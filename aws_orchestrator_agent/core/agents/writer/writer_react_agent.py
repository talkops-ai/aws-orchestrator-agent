"""
Writer React Agent for Terraform module writing and editing.

This module implements the Writer React Agent, which is responsible for
writing, editing, and managing Terraform module files based on generated
content from the generator swarm.

The agent handles:
- Writing Terraform files to disk
- Editing existing Terraform modules
- Managing file operations
- Coordinating with the generator swarm output
"""

from enum import Enum
import os
import traceback
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Annotated
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph.message import add_messages
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent, InjectedState
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.file_management.write import WriteFileTool
from langgraph.graph import StateGraph
from typing_extensions import Annotated

from aws_orchestrator_agent.core.llm.llm_provider import LLMProvider
from aws_orchestrator_agent.config.config import Config
from aws_orchestrator_agent.utils.logger import AgentLogger, log_sync
from aws_orchestrator_agent.core.agents.base_agent import BaseSubgraphAgent
# Create agent logger for writer react agent
writer_react_logger = AgentLogger("WRITER_REACT_AGENT")


class TerraformFileContent(BaseModel):
    """Content for a single Terraform file"""
    file_name: str = Field(description="Terraform file name, e.g. main.tf, variables.tf")
    content: str = Field(description="The HCL content to write to the file")
    file_path: Optional[str] = Field(default="", description="Subdirectory path inside the output directory, optional")


class WriterAgentInput(BaseModel):
    """Input for the writer agent"""
    project_name: str = Field(description="Name of the Terraform project/module")
    output_directory: str = Field(default="./terraform", description="Base output directory for all files")
    files: List[TerraformFileContent] = Field(description="List of files and their content to write")


class WriterAgentOutput(BaseModel):
    """Output from the writer agent"""
    success: bool = Field(description="Whether all files were written successfully")
    files_written: List[str] = Field(description="Absolute or relative paths of all files that were written")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered for individual files")
    summary: str = Field(description="Operation summary")

class WriteStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class OperationType(str, Enum):
    CREATE_DIR = "create_directory"
    VALIDATE = "validate_content"
    WRITE_FILE = "write_file"
    EDIT_FILE = "edit_file"
    BATCH_WRITE = "batch_write"


class FileDescriptor(BaseModel):
    """Describes a file to write."""
    path: str = Field(..., description="Relative path including filename")
    content: str = Field(..., description="HCL or other content")
    subdirectory: Optional[str] = Field(None, description="Optional subdirectory")
    file_type: Optional[str] = Field(None, description="File type (main, variables, outputs, etc.)") 


class FileOperationRecord(BaseModel):
    """Tracks the status of an individual file operation."""
    file: FileDescriptor = Field(..., description="The file being operated on")
    operation: OperationType = Field(..., description="Type of operation")
    status: WriteStatus = Field(WriteStatus.PENDING, description="Operation status")
    started_at: Optional[datetime] = Field(None, description="Timestamp when operation began")
    completed_at: Optional[datetime] = Field(None, description="Timestamp when operation finished")
    error: Optional[str] = Field(None, description="Error message if failed")
    result: Optional[str] = Field(None, description="Operation result message")

class ErrorRecord(BaseModel):
    """Structured error information."""
    code: Optional[str] = Field(None, description="Optional error code")
    message: str = Field(..., description="Error message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    operation_type: Optional[OperationType] = Field(None, description="Operation that caused the error")


class WriterReactState(BaseModel):
    """
    Enhanced state model for the Writer React Agent.
    All fields are optional or have defaults to support handover from other agents.
    """
    # Static input context
    messages: Annotated[List[AnyMessage], add_messages] = Field(
        default_factory=list,
        description="Message history for the writer react workflow"
    )
    
    workspace_path: Optional[str] = Field(None, description="Root workspace directory")
    module_name: Optional[str] = Field(None, description="Terraform module name")
    generation_data: Optional[Dict[str, Any]] = Field(None, description="Generator swarm output context")
    module_structure_plan: Optional[List[Dict[str, Any]]] = Field(None, description="Module structure plan")
    
    # Files management
    files_to_write: List[FileDescriptor] = Field(default_factory=list, description="Files scheduled for writing")
    
    # Operations tracking
    operations: List[FileOperationRecord] = Field(default_factory=list, description="Log of all file operations")
    
    # Agent state
    status: WriteStatus = Field(default=WriteStatus.PENDING, description="Overall writing status")
    current_operation: Optional[OperationType] = Field(default=None, description="Current operation in progress")
    
    # Message management
    internal_messages: List[Any] = Field(default_factory=list, description="Internal agent messages")
    llm_input_messages: List[Any] = Field(default_factory=list, description="Messages sent to LLM")
    
    # Error handling
    errors: List[ErrorRecord] = Field(default_factory=list, description="Structured error records")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings")
    
    # Coordination
    task_id: Optional[str] = Field(default=None, description="Unique task identifier")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    retry_count: int = Field(default=0, description="Number of retries attempted")
    

    # Completion data
    completion_status: Optional[str] = Field(default=None, description="Completion status")
    completion_summary: Optional[str] = Field(default=None, description="Completion summary")
    completion_files_created: Optional[List[str]] = Field(default_factory=list, description="Completion files created")

    # Progress tracking
    progress: Dict[str, Any] = Field(default_factory=dict, description="Progress tracking data")


# State Conversion Functions
def convert_state_to_messages(state: WriterReactState) -> Dict[str, Any]:
    """
    Convert WriterReactState to the format expected by create_react_agent.
    
    Returns a dict with 'messages' key containing LangChain message objects.
    """
    messages = []
    
    # Build comprehensive system message with context
    system_content = f"""You are the Writer React Agent for Terraform module: {state.module_name}

WORKSPACE: {state.workspace_path}
TASK ID: {state.task_id}
SESSION ID: {state.session_id}
CURRENT STATUS: {state.status}
RETRY COUNT: {state.retry_count}

GENERATION DATA CONTEXT:
{json.dumps(state.generation_data, indent=2, default=str)}

FILES TO WRITE ({len(state.files_to_write)}):"""
    
    for i, file_desc in enumerate(state.files_to_write, 1):
        system_content += f"\n{i}. {file_desc.path}"
        if file_desc.subdirectory:
            system_content += f" (in {file_desc.subdirectory})"
        system_content += f" - {file_desc.file_type or 'unknown type'}"
    
    # Add operation history
    if state.operations:
        system_content += f"\n\nPREVIOUS OPERATIONS ({len(state.operations)}):"
        for op in state.operations[-5:]:  # Last 5 operations
            system_content += f"\n- {op.operation}: {op.file.path} - {op.status}"
            if op.error:
                system_content += f" (Error: {op.error})"
    
    # Add errors if any
    if state.errors:
        system_content += f"\n\nPREVIOUS ERRORS ({len(state.errors)}):"
        for error in state.errors[-3:]:  # Last 3 errors
            system_content += f"\n- {error.code}: {error.message}"
    
    # Add warnings
    if state.warnings:
        system_content += f"\n\nWARNINGS:\n" + "\n".join([f"- {warning}" for warning in state.warnings])
    
    messages.append(SystemMessage(content=system_content))
    
    # Add any existing internal messages
    messages.extend(state.internal_messages)
    
    # Add current task message
    if state.current_operation:
        task_message = f"Continue with current operation: {state.current_operation}. Status: {state.status}"
        if state.retry_count > 0:
            task_message += f" (Retry attempt: {state.retry_count})"
    else:
        task_message = "Please analyze the generation data and begin writing the Terraform files to the workspace."
    
    messages.append(HumanMessage(content=task_message))
    
    return {"messages": messages}


def update_state_from_messages(state: WriterReactState, messages: List[Any]) -> WriterReactState:
    """
    Update WriterReactState with results from agent execution.
    
    Args:
        state: Current WriterReactState
        messages: Messages returned from agent execution
        
    Returns:
        Updated WriterReactState
    """
    # Store the conversation in internal_messages
    state.internal_messages = messages
    
    # Extract information from the last AI message
    if messages:
        last_message = messages[-1]
        if hasattr(last_message, 'content') and last_message.content:
            content = last_message.content.lower()
            
            # Update status based on message content
            if "completed" in content or "finished" in content:
                state.status = WriteStatus.COMPLETED
                state.current_operation = None
            elif "failed" in content or "error" in content:
                state.status = WriteStatus.FAILED
            elif "writing" in content or "creating" in content:
                state.status = WriteStatus.IN_PROGRESS
    
    # Update status based on operations
    if state.operations:
        completed_ops = [op for op in state.operations if op.status == WriteStatus.COMPLETED]
        failed_ops = [op for op in state.operations if op.status == WriteStatus.FAILED]
        
        if len(completed_ops) == len(state.files_to_write) and not failed_ops:
            state.status = WriteStatus.COMPLETED
        elif failed_ops:
            state.status = WriteStatus.FAILED
    
    return state


class TerraformWriterAgent:
    """
    Writer agent using LangChain's WriteFileTool to write HCL files to disk.
    """
    
    def __init__(self, root_directory: str = "./terraform"):
        """
        Initialize the writer agent.
        
        Args:
            root_directory: Base directory where all files will be written
        """
        self.root_directory = root_directory
        self.write_tool = WriteFileTool(root_dir=root_directory)

    def write_terraform_files(self, input_data: WriterAgentInput) -> WriterAgentOutput:
        """
        Write all Terraform files to disk using WriteFileTool.
        
        Args:
            input_data: WriterAgentInput containing files to write
            
        Returns:
            WriterAgentOutput with results of write operations
        """
        written_files = []
        errors = []

        # Ensure base output directory exists
        output_dir = os.path.join(self.root_directory, input_data.output_directory.lstrip("./"))
        os.makedirs(output_dir, exist_ok=True)

        for tf_file in input_data.files:
            try:
                # Construct target file path within root/output dir
                if tf_file.file_path:
                    full_dir = os.path.join(output_dir, tf_file.file_path)
                    os.makedirs(full_dir, exist_ok=True)
                    file_path = os.path.join(full_dir, tf_file.file_name)
                else:
                    file_path = os.path.join(output_dir, tf_file.file_name)

                # Use WriteFileTool to write the file
                result = self.write_tool.invoke({
                    "file_path": file_path,
                    "text": tf_file.content
                })
                
                written_files.append(file_path)
                writer_react_logger.log_structured(
                    level="INFO",
                    message="Successfully wrote Terraform file",
                    extra={"file_path": file_path}
                )

            except ValidationError as ve:
                error_msg = f"Validation error for {tf_file.file_name}: {str(ve)}"
                errors.append(error_msg)
                writer_react_logger.log_structured(
                    level="ERROR",
                    message="Validation error writing file",
                    extra={"error": error_msg, "file_name": tf_file.file_name}
                )
                
            except Exception as e:
                error_msg = f"Failed to write {tf_file.file_name}: {str(e)}"
                errors.append(error_msg)
                writer_react_logger.log_structured(
                    level="ERROR",
                    message="Failed to write file",
                    extra={"error": error_msg, "file_name": tf_file.file_name}
                )

        # Create summary
        total_files = len(input_data.files)
        success_count = len(written_files)
        error_count = len(errors)
        success = error_count == 0
        
        summary = f"Writer Agent: {success_count}/{total_files} files written successfully."
        if errors:
            summary += f" {error_count} errors encountered."

        return WriterAgentOutput(
            success=success,
            files_written=written_files,
            errors=errors,
            summary=summary
        )

    def validate_input(self, input_data: WriterAgentInput) -> List[str]:
        """
        Validate input data before processing.
        
        Args:
            input_data: Input to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        validation_errors = []
        
        if not input_data.files:
            validation_errors.append("No files provided to write")
            
        for i, tf_file in enumerate(input_data.files):
            if not tf_file.file_name:
                validation_errors.append(f"File {i}: file_name is required")
            if not tf_file.content:
                validation_errors.append(f"File {i}: content is required")
                
        return validation_errors


class WriterReactAgent(BaseSubgraphAgent):
    """
    Writer React Agent that handles Terraform module writing and editing.
    
    This agent is responsible for:
    - Writing generated Terraform content to files
    - Editing existing Terraform modules
    - Managing file operations and workspace coordination
    - Coordinating with generator swarm output
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        custom_config: Optional[Dict[str, Any]] = None,
        name: str = "writer_react_agent",
        memory: Optional[MemorySaver] = None
    ):
        """
        Initialize the Writer React Agent.
        
        Args:
            config: Configuration instance (defaults to new Config())
            custom_config: Optional custom configuration to override defaults
            name: Agent name for identification
            memory: Shared memory/checkpointer instance
        """
        writer_react_logger.log_structured(
            level="INFO",
            message="=== WRITER REACT AGENT INITIALIZATION START ===",
            extra={
                "name": name,
                "has_config": config is not None,
                "has_custom_config": custom_config is not None,
                "has_memory": memory is not None
            }
        )
        
        # Use centralized config system
        self.config_instance = config or Config(custom_config or {})
        
        # Set agent name for identification
        self._name = name
        
        # Set shared memory
        self.memory = memory or MemorySaver()
        
        writer_react_logger.log_structured(
            level="DEBUG",
            message="Basic initialization complete",
            extra={
                "config_type": type(self.config_instance).__name__,
                "memory_type": type(self.memory).__name__
            }
        )
        
        # Get LLM configuration from centralized config
        llm_config = self.config_instance.get_llm_config()
        
        writer_react_logger.log_structured(
            level="DEBUG",
            message="LLM config retrieved",
            extra={
                "llm_provider": llm_config.get('provider'),
                "llm_model": llm_config.get('model'),
                "llm_temperature": llm_config.get('temperature'),
                "llm_max_tokens": llm_config.get('max_tokens')
            }
        )
        
        # Initialize the LLM model using the centralized provider
        try:
            llm_higher_config = self.config_instance.get_llm_higher_config()
            self.model = LLMProvider.create_llm(
                provider=llm_higher_config['provider'],
                model=llm_higher_config['model'],
                temperature=llm_higher_config['temperature'],
                max_tokens=llm_higher_config['max_tokens']
            )

            writer_react_logger.log_structured(
                level="INFO",
                message="LLM model initialized successfully",
                extra={
                    "llm_provider": llm_config['provider'], 
                    "llm_model": llm_config['model'],
                    "model_type": type(self.model).__name__
                }
            )
        except Exception as e:
            writer_react_logger.log_structured(
                level="ERROR",
                message="LLM model initialization failed",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "llm_provider": llm_config.get('provider'),
                    "llm_model": llm_config.get('model')
                }
            )
            raise
        
        writer_react_logger.log_structured(
            level="INFO",
            message="=== WRITER REACT AGENT INITIALIZATION COMPLETE ===",
            extra={
                "llm_provider": llm_config['provider'],
                "llm_model": llm_config['model'],
                "name": name
            }
        )
    
    def _process_generation_data(self, state: WriterReactState) -> WriterReactState:
        """
        Process generation data and populate files_to_write list.
        Handles handover scenarios where generation_data might be None or incomplete.
        
        Args:
            state: Current writer react state
            
        Returns:
            Updated state with files_to_write populated
        """
        if not state.generation_data:
            writer_react_logger.log_structured(
                level="DEBUG",
                message="No generation data to process",
                extra={"task_id": state.task_id}
            )
            return state
            
        files_to_write = []
        
        writer_react_logger.log_structured(
            level="DEBUG",
            message="Processing generation data",
            extra={
                "generation_data_keys": list(state.generation_data.keys()) if isinstance(state.generation_data, dict) else "non_dict",
                "task_id": state.task_id
            }
        )
        
        # Extract different file types from generation data
        if "terraform_files" in state.generation_data:
            for file_info in state.generation_data["terraform_files"]:
                file_desc = FileDescriptor(
                    path=file_info.get("file_name", "unknown.tf"),
                    content=file_info.get("content", ""),
                    subdirectory=file_info.get("subdirectory", ""),
                    file_type=file_info.get("file_type", "unknown")
                )
                files_to_write.append(file_desc)
        
        # Handle legacy format with generated_module
        elif "generated_module" in state.generation_data:
            module_data = state.generation_data["generated_module"]
            
            # Process each type of generated content
            if module_data.get("resources"):
                files_to_write.append(FileDescriptor(
                    path="main.tf",
                    content=module_data["resources"],
                    subdirectory=None,
                    file_type="main"
                ))
            
            if module_data.get("variables"):
                files_to_write.append(FileDescriptor(
                    path="variables.tf",
                    content=module_data["variables"],
                    subdirectory=None,
                    file_type="variables"
                ))
            
            if module_data.get("outputs"):
                files_to_write.append(FileDescriptor(
                    path="outputs.tf",
                    content=module_data["outputs"],
                    subdirectory=None,
                    file_type="outputs"
                ))
            
            if module_data.get("data_sources"):
                files_to_write.append(FileDescriptor(
                    path="data.tf",
                    content=module_data["data_sources"],
                    subdirectory=None,
                    file_type="data"
                ))
            
            if module_data.get("locals"):
                files_to_write.append(FileDescriptor(
                    path="locals.tf",
                    content=module_data["locals"],
                    subdirectory=None,
                    file_type="locals"
                ))
            
            if module_data.get("backend"):
                files_to_write.append(FileDescriptor(
                    path="backend.tf",
                    content=module_data["backend"],
                    subdirectory=None,
                    file_type="backend"
                ))
            
            if module_data.get("readme"):
                files_to_write.append(FileDescriptor(
                    path="README.md",
                    content=module_data["readme"],
                    subdirectory=None,
                    file_type="readme"
                ))
        
        # Handle other formats
        for key, content in state.generation_data.items():
            if key.endswith("_content") and isinstance(content, str):
                file_name = key.replace("_content", ".tf")
                file_desc = FileDescriptor(
                    path=file_name,
                    content=content,
                    file_type=key.replace("_content", "")
                )
                files_to_write.append(file_desc)
        
        # Update state with processed files
        state.files_to_write = files_to_write
        state.status = WriteStatus.IN_PROGRESS
        
        writer_react_logger.log_structured(
            level="INFO",
            message="Processed generation data and created files_to_write list",
            extra={
                "files_count": len(files_to_write),
                "file_names": [f.path for f in files_to_write],
                "file_types": [f.file_type for f in files_to_write]
            }
        )
        
        return state
    
    def output_transform(self, agent_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform agent state back to supervisor state using StateTransformer.
        
        Args:
            agent_state: The final state from agent execution
        
        Returns:
            Dict[str, Any]: Data to merge into supervisor state
        """
        try:
            writer_react_logger.log_structured(
                level="INFO",
                message="Transforming writer state back to supervisor state using StateTransformer",
                extra={
                    "agent_state_type": type(agent_state).__name__,
                    "status": getattr(agent_state, "status", "unknown"),
                    "operations_count": len(getattr(agent_state, "operations", []))
                }
            )
            
            # Use StateTransformer to convert writer state to supervisor updates
            from aws_orchestrator_agent.core.agents.types import StateTransformer
            supervisor_updates = StateTransformer.writer_to_supervisor(agent_state)
            
            writer_react_logger.log_structured(
                level="INFO",
                message="Successfully transformed writer state to supervisor updates",
                extra={
                    "supervisor_updates_keys": list(supervisor_updates.keys()),
                    "completion_status": supervisor_updates.get("completion_status", "unknown"),
                    "files_created_count": len(supervisor_updates.get("completion_files_created", []))
                }
            )
            
            return supervisor_updates
            
        except Exception as e:
            writer_react_logger.log_structured(
                level="ERROR",
                message="Failed to transform writer state to supervisor updates",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc()
                }
            )
            # Return minimal fallback updates
            return {
                "completion_status": "failed",
                "completion_summary": f"Writer agent failed: {str(e)}",
                "completion_files_created": [],
                "status": "failed"
            }
    
    @property
    def name(self) -> str:
        """Agent name for Send() routing and identification."""
        return self._name
    
    @property
    def state_model(self) -> type[BaseModel]:
        """Get the state model for this agent."""
        return WriterReactState
    
    def _create_writer_agent(self):
        """Create the main Writer React Agent."""
        writer_react_logger.log_structured(
            level="DEBUG",
            message="Creating writer react agent",
            extra={}
        )
        
        return create_react_agent(
            model=self.model,
            tools=[
                self._create_write_terraform_file_tool(),
                self._create_batch_write_terraform_files_tool(),
                self._create_edit_terraform_file_tool(),
                self._create_validate_terraform_tool(),
                self._create_create_directory_tool(),
                self._create_list_files_tool(),
                self._create_read_file_tool(),
                self._create_completion_tool()
            ],
            name="writer_react_agent",
            prompt=ChatPromptTemplate.from_messages([
                ("system", """
You are **Writer React Agent**, the final executor in a multi-agent Terraform provisioning pipeline.
Your mission: take **generation_data** (Terraform HCL content) and create a structured module on disk.

EXECUTION PHASES:
1. **Analyze**: Review generation data and files to write
2. **Plan**: Determine directory structure and file operations
3. **Create Directories**: Use create_directory tool for any needed folders
4. **Validate Content**: Use validate_terraform_syntax before writing
5. **Write Files**: Use write_terraform_file or batch_write_terraform_files
6. **Complete**: Use completion_tool to finalize and report results

TOOLS AVAILABLE:
- create_directory({{"path": "subdirectory/path"}}) 
- validate_terraform_syntax({{"content": "HCL content"}})
- write_terraform_file({{"file_path": "path/file.tf", "text": "content"}})
- batch_write_terraform_files({{"files": [{{"file_path": "...", "text": "..."}}, ...]}})
- edit_terraform_file({{"file_path": "path/file.tf", "new_content": "content"}})
- list_files({{"directory_path": "path", "pattern": "*.tf"}})
- read_file({{"file_path": "path/file.tf"}})
- completion_tool({{"result": "completed/failed", "summary": "optional details", "files_created": ["file1.tf", "file2.tf"]}})

GUIDELINES:
- Always validate syntax before writing files
- Create necessary directories first
- Use batch operations for efficiency when possible
- Handle errors gracefully and continue with remaining operations
- Always call completion_tool at the end with a summary

BEGIN: Analyze the provided generation data and execute the file writing workflow."""),
                MessagesPlaceholder(variable_name="messages")
            ])
        )
    
    def _create_write_terraform_file_tool(self):
        """Create tool for writing Terraform files using WriteFileTool with state injection."""
        
        @tool
        def write_terraform_file(
            file_data: dict,
            state: Annotated[WriterReactState, InjectedState]
        ) -> str:
            """
            Write a Terraform file to disk with state tracking.
            
            Args:
                file_data: Dict with 'file_path' and 'text' keys
                state: Injected WriterReactState (invisible to LLM)
            """
            try:
                # Access rich state data
                workspace = state.workspace_path or "./"
                module_name = state.module_name
                
                writer_react_logger.log_structured(
                    level="DEBUG",
                    message="Starting file write operation",
                    extra={
                        "file_path": file_data["file_path"],
                        "workspace": workspace,
                        "module_name": module_name,
                        "content_length": len(file_data.get("text", ""))
                    }
                )
                
                # Create operation record
                operation = FileOperationRecord(
                    file=FileDescriptor(
                        path=file_data["file_path"],
                        content=file_data.get("text", "")
                    ),
                    operation=OperationType.WRITE_FILE,
                    status=WriteStatus.IN_PROGRESS,
                    started_at=datetime.now(timezone.utc)
                )
                state.operations.append(operation)
                state.current_operation = OperationType.WRITE_FILE
                
                # Use WriteFileTool
                write_tool = WriteFileTool(root_dir=workspace)
                result = write_tool.invoke({
                    "file_path": file_data["file_path"],
                    "text": file_data.get("text", "")
                })
                
                # Update operation record
                operation.status = WriteStatus.COMPLETED
                operation.completed_at = datetime.now(timezone.utc)
                operation.result = result
                
                writer_react_logger.log_structured(
                    level="INFO",
                    message="Successfully wrote Terraform file",
                    extra={
                        "file_path": file_data["file_path"],
                        "content_length": len(file_data.get("text", "")),
                        "workspace": workspace
                    }
                )
                
                return f"Successfully wrote file: {file_data['file_path']} ({len(file_data.get('text', ''))} bytes)"
                
            except Exception as e:
                # Log error to state
                error = ErrorRecord(
                    code="WRITE_FILE_ERROR",
                    message=str(e),
                    operation_type=OperationType.WRITE_FILE
                )
                state.errors.append(error)
                
                writer_react_logger.log_structured(
                    level="ERROR",
                    message="Failed to write Terraform file",
                    extra={
                        "file_path": file_data.get("file_path", "unknown"),
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "workspace": state.workspace_path
                    }
                )
                
                # Update operation record
                if state.operations:
                    state.operations[-1].status = WriteStatus.FAILED
                    state.operations[-1].error = str(e)
                    state.operations[-1].completed_at = datetime.now(timezone.utc)
                
                return f"Failed to write file {file_data.get('file_path', 'unknown')}: {str(e)}"
        
        return write_terraform_file
    
    def _create_batch_write_terraform_files_tool(self):
        """Create tool for batch writing multiple Terraform files with state injection."""
        
        @tool
        def batch_write_terraform_files(
            files_data: dict,
            state: Annotated[WriterReactState, InjectedState]
        ) -> str:
            """
            Write multiple Terraform files in batch with state tracking.
            
            Args:
                files_data: Dict with 'files' key containing list of file data
                state: Injected WriterReactState
            """
            files_list = files_data.get("files", [])
            results = []
            success_count = 0
            
            writer_react_logger.log_structured(
                level="DEBUG",
                message="Starting batch write operation",
                extra={
                    "files_count": len(files_list),
                    "workspace": state.workspace_path,
                    "module_name": state.module_name
                }
            )
            
            for file_data in files_list:
                try:
                    # Create operation record for each file
                    operation = FileOperationRecord(
                        file=FileDescriptor(
                            path=file_data.get("file_path", ""),
                            content=file_data.get("text", "")
                        ),
                        operation=OperationType.BATCH_WRITE,
                        status=WriteStatus.IN_PROGRESS,
                        started_at=datetime.now(timezone.utc)
                    )
                    state.operations.append(operation)
                    
                    writer_react_logger.log_structured(
                        level="DEBUG",
                        message="Created operation record for batch write",
                        extra={
                            "file_path": file_data.get("file_path", ""),
                            "operations_count": len(state.operations),
                            "task_id": state.task_id
                        }
                    )
                    
                    # Use WriteFileTool for each file
                    workspace = state.workspace_path or "./"
                    write_tool = WriteFileTool(root_dir=workspace)
                    result = write_tool.invoke({
                        "file_path": file_data.get("file_path", ""),
                        "text": file_data.get("text", "")
                    })
                    
                    # Update operation record
                    operation.status = WriteStatus.COMPLETED
                    operation.completed_at = datetime.now(timezone.utc)
                    operation.result = result
                    success_count += 1
                    results.append(f"✅ {file_data.get('file_path', 'unknown')}")
                    
                except Exception as e:
                    # Log error to state
                    error = ErrorRecord(
                        code="BATCH_WRITE_ERROR",
                        message=str(e),
                        operation_type=OperationType.BATCH_WRITE
                    )
                    state.errors.append(error)
                    
                    # Update operation record
                    if state.operations:
                        state.operations[-1].status = WriteStatus.FAILED
                        state.operations[-1].error = str(e)
                        state.operations[-1].completed_at = datetime.now(timezone.utc)
                    
                    results.append(f"❌ {file_data.get('file_path', 'unknown')}: {str(e)}")
            
            summary = f"Batch write completed: {success_count}/{len(files_list)} files written successfully"
            
            writer_react_logger.log_structured(
                level="INFO",
                message="Batch write operation completed",
                extra={
                    "success_count": success_count,
                    "total_files": len(files_list),
                    "failed_count": len(files_list) - success_count,
                    "workspace": state.workspace_path
                }
            )
            
            return summary + "\n" + "\n".join(results)
        
        return batch_write_terraform_files
    
    def _create_edit_terraform_file_tool(self):
        """Create tool for editing existing Terraform files with state injection."""
        
        @tool
        def edit_terraform_file(
            edit_data: dict,
            state: Annotated[WriterReactState, InjectedState]
        ) -> str:
            """
            Edit an existing Terraform file with state tracking.
            
            Args:
                edit_data: Dict with 'file_path', 'new_content', and optional 'backup' keys
                state: Injected WriterReactState
            """
            try:
                file_path = edit_data["file_path"]
                new_content = edit_data.get("new_content", "")
                backup = edit_data.get("backup", True)
                
                # Create operation record
                operation = FileOperationRecord(
                    file=FileDescriptor(
                        path=file_path,
                        content=new_content
                    ),
                    operation=OperationType.EDIT_FILE,
                    status=WriteStatus.IN_PROGRESS,
                    started_at=datetime.now(timezone.utc)
                )
                state.operations.append(operation)
                state.current_operation = OperationType.EDIT_FILE
                
                # Create backup if requested
                backup_created = False
                if backup and Path(file_path).exists():
                    backup_path = f"{file_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    Path(file_path).rename(backup_path)
                    backup_created = True
                
                # Write new content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                # Update operation record
                operation.status = WriteStatus.COMPLETED
                operation.completed_at = datetime.now(timezone.utc)
                operation.result = f"File edited successfully, backup: {backup_created}"
                
                return f"Successfully edited file: {file_path} ({len(new_content)} bytes, backup: {backup_created})"
                
            except Exception as e:
                # Log error to state
                error = ErrorRecord(
                    code="EDIT_FILE_ERROR",
                    message=str(e),
                    operation_type=OperationType.EDIT_FILE
                )
                state.errors.append(error)
                
                # Update operation record
                if state.operations:
                    state.operations[-1].status = WriteStatus.FAILED
                    state.operations[-1].error = str(e)
                    state.operations[-1].completed_at = datetime.now(timezone.utc)
                
                return f"Failed to edit file {edit_data.get('file_path', 'unknown')}: {str(e)}"
        
        return edit_terraform_file
    
    def _create_validate_terraform_tool(self):
        """Create tool for validating Terraform files with state injection."""
        
        @tool
        def validate_terraform_syntax(
            validation_data: dict,
            state: Annotated[WriterReactState, InjectedState]
        ) -> str:
            """
            Validate Terraform syntax with state tracking.
            
            Args:
                validation_data: Dict with 'content' key containing HCL to validate
                state: Injected WriterReactState
            """
            try:
                content = validation_data.get("content", "")
                
                if not content.strip():
                    state.warnings.append("Empty content provided for validation")
                    return "Warning: Empty content - nothing to validate"
                
                # Basic validation
                terraform_keywords = ["resource", "variable", "output", "data", "locals", "module"]
                found_keywords = [kw for kw in terraform_keywords if kw in content]
                
                if not found_keywords:
                    warning = "No Terraform keywords found in content"
                    state.warnings.append(warning)
                    return f"Warning: {warning}"
                
                # Check for basic syntax issues
                if content.count("{") != content.count("}"):
                    error = ErrorRecord(
                        code="SYNTAX_ERROR",
                        message="Mismatched braces in Terraform content",
                        operation_type=OperationType.VALIDATE
                    )
                    state.errors.append(error)
                    return "Error: Mismatched braces detected"
                
                return f"Syntax validation passed. Found keywords: {', '.join(found_keywords)}"
                
            except Exception as e:
                error = ErrorRecord(
                    code="VALIDATION_ERROR",
                    message=str(e),
                    operation_type=OperationType.VALIDATE
                )
                state.errors.append(error)
                return f"Validation failed: {str(e)}"
        
        return validate_terraform_syntax
    
    def _create_create_directory_tool(self):
        """Create tool for creating directories with state injection."""
        
        @tool
        def create_directory(
            directory_data: dict,
            state: Annotated[WriterReactState, InjectedState]
        ) -> str:
            """
            Create a directory with state tracking.
            
            Args:
                directory_data: Dict with 'path' key
                state: Injected WriterReactState
            """
            try:
                dir_path = directory_data["path"]
                workspace = state.workspace_path or "./"
                full_path = os.path.join(workspace, dir_path.lstrip("./"))
                
                # Create operation record
                operation = FileOperationRecord(
                    file=FileDescriptor(path=dir_path, content="<directory>"),
                    operation=OperationType.CREATE_DIR,
                    status=WriteStatus.IN_PROGRESS,
                    started_at=datetime.now(timezone.utc)
                )
                state.operations.append(operation)
                
                # Create directory
                os.makedirs(full_path, exist_ok=True)
                
                # Update operation
                operation.status = WriteStatus.COMPLETED
                operation.completed_at = datetime.now(timezone.utc)
                operation.result = f"Directory created: {full_path}"
                
                return f"Successfully created directory: {dir_path}"
                
            except Exception as e:
                error = ErrorRecord(
                    code="CREATE_DIR_ERROR", 
                    message=str(e),
                    operation_type=OperationType.CREATE_DIR
                )
                state.errors.append(error)
                
                if state.operations:
                    state.operations[-1].status = WriteStatus.FAILED
                    state.operations[-1].error = str(e)
                
                return f"Failed to create directory {directory_data.get('path', 'unknown')}: {str(e)}"
        
        return create_directory
    
    def _create_list_files_tool(self):
        """Create tool for listing files in a directory with state injection."""
        
        @tool
        def list_files(
            list_data: dict,
            state: Annotated[WriterReactState, InjectedState]
        ) -> str:
            """
            List files in a directory with state tracking.
            
            Args:
                list_data: Dict with 'directory_path' and optional 'pattern' keys
                state: Injected WriterReactState
            """
            try:
                directory_path = list_data["directory_path"]
                pattern = list_data.get("pattern", "*.tf")
                
                directory = Path(directory_path)
                if not directory.exists():
                    warning = f"Directory {directory_path} does not exist"
                    state.warnings.append(warning)
                    return f"Warning: {warning}"
                
                files = list(directory.glob(pattern))
                file_list = [str(f) for f in files]
                
                return f"Found {len(file_list)} files matching '{pattern}' in {directory_path}:\n" + "\n".join(file_list)
                
            except Exception as e:
                error = ErrorRecord(
                    code="LIST_FILES_ERROR",
                    message=str(e)
                )
                state.errors.append(error)
                return f"Failed to list files in {list_data.get('directory_path', 'unknown')}: {str(e)}"
        
        return list_files
    
    def _create_read_file_tool(self):
        """Create tool for reading file contents with state injection."""
        
        @tool
        def read_file(
            read_data: dict,
            state: Annotated[WriterReactState, InjectedState]
        ) -> str:
            """
            Read the contents of a file with state tracking.
            
            Args:
                read_data: Dict with 'file_path' key
                state: Injected WriterReactState
            """
            try:
                file_path = read_data["file_path"]
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                return f"Successfully read file: {file_path} ({len(content)} characters)\nContent:\n{content[:500]}{'...' if len(content) > 500 else ''}"
                
            except Exception as e:
                error = ErrorRecord(
                    code="READ_FILE_ERROR",
                    message=str(e)
                )
                state.errors.append(error)
                return f"Failed to read file {read_data.get('file_path', 'unknown')}: {str(e)}"
        
        return read_file
    
    def _safe_get_state_attr(self, state, attr_name, default=None):
        """Safely get an attribute from state, handling cases where state injection fails."""
        if hasattr(state, attr_name):
            return getattr(state, attr_name)
        return default
    
    
    def _create_completion_tool(self):
        """Create tool for marking task completion with state injection and handover."""
        
        @tool
        def completion_tool(
            completion_data: dict,
            state: Annotated[WriterReactState, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId]
        ) -> dict:
            """
            Mark the writing process as complete, transform state, and handover to supervisor.
            
            Args:
                completion_data: Dict with 'result', optional 'summary', and optional 'files_created' keys
                state: Injected WriterReactState
                tool_call_id: Injected tool call ID for proper message handling
            """
            result = completion_data.get("result", "completed")
            summary = completion_data.get("summary", "")
            files_created = completion_data.get("files_created", [])
            
            # Log completion
            writer_react_logger.log_structured(
                level="INFO",
                message="Writer agent completion tool called",
                extra={
                    "result": result,
                    "has_summary": bool(summary),
                    "files_created_count": len(files_created),
                    "files_created": files_created
                }
            )
            
            # # Create completion tool message for supervisor
            # completion_tool_message = ToolMessage(
            #     content=f"Writer agent completion - {result}. Files created: {len(files_created)}. Summary: {summary[:100] + '...' if len(summary) > 100 else summary}",
            #     name="writer_complete_task",
            #     tool_call_id=tool_call_id,
            #     metadata={
            #         "completion_type": "writer_complete",
            #         "result": result,
            #         "files_created": files_created,
            #         "summary": summary,
            #         "files_created_count": len(files_created)
            #     }
            # )
            # writer_completion_msg = HumanMessage(content=f"Writer agent execution completed successfully and has written the following files: {files_created}")
            # supervisor_handover_data = {
            #     "status": "completed" if result == "completed" else "failed",
            #     "summary": summary,
            #     "files_created": files_created
            # }
            # Return state updates instead of Command (since we're running as a subgraph)
            # The supervisor will handle the state transformation
            return {
                "completion_status": result,
                "completion_summary": summary,
                "completion_files_created": files_created,
                "status": "completed" if result == "completed" else "failed"
            }
        
        return completion_tool
    
    def build_graph(self) -> StateGraph:
        """
        Build the LangGraph StateGraph for the writer react agent with proper state management.
        
        Returns:
            StateGraph: The compiled graph for this agent
        """
        try:
            writer_react_logger.log_structured(
                level="INFO",
                message="=== WRITER REACT AGENT BUILD GRAPH START ===",
                extra={
                    "agent_name": getattr(self, '_name', 'unknown'),
                    "model_initialized": hasattr(self, 'model')
                }
            )
            
            def preprocess_generation_data(state: WriterReactState) -> WriterReactState:
                """Preprocess generation data and setup for agent execution."""
                writer_react_logger.log_structured(
                    level="DEBUG",
                    message="Starting preprocessing of generation data",
                    extra={
                        "has_generation_data": bool(state.generation_data),
                        "task_id": state.task_id,
                        "session_id": state.session_id
                    }
                )
                
                # Initialize workspace_path from config if not set
                if not state.workspace_path:
                    try:
                        module_path = self.config_instance.MODULE_PATH
                        if module_path:
                            state.workspace_path = module_path
                            writer_react_logger.log_structured(
                                level="DEBUG",
                                message="Initialized workspace_path from config in preprocessing",
                                extra={
                                    "workspace_path": state.workspace_path,
                                    "task_id": state.task_id
                                }
                            )
                        else:
                            state.workspace_path = "./"
                            writer_react_logger.log_structured(
                                level="WARNING",
                                message="MODULE_PATH not found in config, using default",
                                extra={"task_id": state.task_id}
                            )
                    except AttributeError:
                        state.workspace_path = "./"
                        writer_react_logger.log_structured(
                            level="WARNING",
                            message="MODULE_PATH not found in config, using default",
                            extra={"task_id": state.task_id}
                        )
                    except Exception as e:
                        state.workspace_path = "./"
                        writer_react_logger.log_structured(
                            level="WARNING",
                            message="Failed to get MODULE_PATH from config, using default",
                            extra={"error": str(e), "task_id": state.task_id}
                        )
                
                processed_state = self._process_generation_data(state)
                
                # Convert to messages format
                messages_data = convert_state_to_messages(processed_state)
                processed_state.llm_input_messages = messages_data["messages"]
                
                writer_react_logger.log_structured(
                    level="DEBUG",
                    message="State converted to LLM messages",
                    extra={
                        "messages_count": len(processed_state.llm_input_messages),
                        "task_id": state.task_id
                    }
                )
                
                return processed_state
            
            async def run_writer_agent(state: WriterReactState) -> Dict[str, Any]:
                """Execute the writer agent with state injection."""
                writer_react_logger.log_structured(
                    level="DEBUG",
                    message="Starting writer agent execution",
                    extra={
                        "messages_count": len(state.llm_input_messages),
                        "task_id": state.task_id,
                        "session_id": state.session_id
                    }
                )
                
                writer_agent = self._create_writer_agent()
                
                # Prepare agent input - pass the full state as a dict so InjectedState can access it
                agent_input = state.model_dump()
                # Ensure messages are in the right format for the agent
                agent_input["messages"] = state.llm_input_messages
                
                try:
                    # Execute agent (tools will receive injected state) with increased recursion limit
                    result = await writer_agent.ainvoke(
                        agent_input,
                        config={"recursion_limit": 40}  # Increase from default (usually 25)
                    )
                    
                    # parse last 3 messages from the result and get the completion data
                    last_3_messages = result.get("messages", [])[-3:]
                    response = {}
                    for message in last_3_messages:
                        if isinstance(message, ToolMessage):
                            content = message.content
                            if isinstance(content, str):
                                response = json.loads(content)
                                break

                    completion_status = response.get("completion_status", "completed")
                    completion_summary = response.get("completion_summary", "")
                    completion_files_created = response.get("completion_files_created", [])
                    state.completion_status = completion_status
                    state.completion_summary = completion_summary
                    state.completion_files_created = completion_files_created
                    state.status = WriteStatus.COMPLETED if completion_status == "completed" else WriteStatus.FAILED
                    # state.messages = result.get("messages", [])
                    # Use StateTransformer to convert completion data to supervisor updates
                    from aws_orchestrator_agent.core.agents.types import StateTransformer
                    supervisor_updates = StateTransformer.writer_to_supervisor(state.model_dump())
                    supervisor_updates["messages"] = result.get("messages", [])
                    
                    writer_react_logger.log_structured(
                        level="INFO",
                        message="Writer agent: Successfully processed state",
                        extra={
                            "supervisor_updates_keys": list(supervisor_updates.keys())
                        }
                    )
                    
                    return supervisor_updates
                    
                except Exception as e:
                    # Handle agent execution errors
                    error = ErrorRecord(
                        code="AGENT_EXECUTION_ERROR",
                        message=str(e)
                    )
                    state.errors.append(error)
                    state.status = WriteStatus.FAILED
                    
                    writer_react_logger.log_structured(
                        level="ERROR",
                        message="Writer agent execution failed",
                        extra={
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "task_id": state.task_id,
                            "session_id": state.session_id
                        }
                    )
                    return {
                        "messages": state.messages,
                        "writer_data": {
                            "status": WriteStatus.FAILED.value,
                            "summary": str(e),
                            "files_created": []
                        },
                        "llm_input_messages": [HumanMessage(content=f"Writer agent execution failed: {str(e)}")]
                    }
                
            # Create the graph
            graph = StateGraph(WriterReactState)
            
            # Add nodes
            graph.add_node("preprocess_data", preprocess_generation_data)
            graph.add_node("writer_agent", run_writer_agent)
            
            # Setup flow
            graph.set_entry_point("preprocess_data")
            graph.add_edge("preprocess_data", "writer_agent")
            graph.set_finish_point("writer_agent")
            
            writer_react_logger.log_structured(
                level="INFO",
                message="=== WRITER REACT AGENT BUILD GRAPH COMPLETE ===",
                extra={
                    "graph_type": type(graph).__name__,
                    "nodes": ["preprocess_data", "writer_agent"]
                }
            )
            
            return graph
            
        except Exception as e:
            writer_react_logger.log_structured(
                level="ERROR",
                message="=== WRITER REACT AGENT BUILD GRAPH FAILED ===",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc()
                }
            )
            raise


# Factory function for easy creation
@log_sync
def create_writer_react_agent(
    config: Optional[Config] = None,
    custom_config: Optional[Dict[str, Any]] = None,
    name: str = "writer_react_agent"
) -> WriterReactAgent:
    """
    Factory function to create a Writer React Agent.
    
    Args:
        config: Configuration instance
        custom_config: Optional custom configuration
        name: Agent name
        
    Returns:
        Configured WriterReactAgent instance
    """
    return WriterReactAgent(config=config, custom_config=custom_config, name=name)

