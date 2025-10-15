"""
Custom Supervisor Agent using langgraph-supervisor.

This module implements a supervisor agent that coordinates specialized
agents using langgraph-supervisor's create_supervisor() function.
Uses centralized Config and LLMProvider.
"""

import logging
import uuid
import ast
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Callable, AsyncGenerator, Annotated
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Send, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
import asyncio
from contextlib import asynccontextmanager

# Import langgraph-supervisor
from langgraph_supervisor import create_supervisor

# Import our custom handoff tools
from .supervisor_handoff_tools import create_handoff_tools_for_agents

from .types import (
    SupervisorState, 
    SupervisorWorkflowState,
    AgentType, 
    get_state_class, 
    WorkflowStatus, 
    BaseAgent, 
    AgentResponse,
    StateTransformer,
    GenerationState,
    ValidationState,
    EditorState,
    SecurityState,
    CostState
)
from .generator.generator_state import GeneratorSwarmState
from .base_agent import BaseSubgraphAgent
from aws_orchestrator_agent.utils.logger import AgentLogger, log_sync, log_async
from aws_orchestrator_agent.config.config import Config
from aws_orchestrator_agent.core.llm.llm_provider import LLMProvider

# Create agent logger for supervisor
supervisor_logger = AgentLogger("SUPERVISOR")

@asynccontextmanager
async def isolation_shield():
    """Shield sub-supervisor from parent cancellation"""
    try:
        yield
    except asyncio.CancelledError:
        supervisor_logger.log_structured(
            level="WARNING",
            message="Parent cancelled, but allowing sub-supervisor to complete",
            extra={
                "current_task": str(asyncio.current_task()) if asyncio.current_task() else "No current task",
                "timestamp": datetime.now().isoformat()
            }
        )
        # Don't re-raise immediately, allow graceful completion
        await asyncio.sleep(0.1)  # Brief delay for cleanup
        raise


class CustomSupervisorAgent(BaseAgent):
    """
    Custom supervisor agent that orchestrates subgraph agents.
    
    This supervisor:
    1. Manages workflow state and routing decisions
    2. Delegates tasks to specialized agents using Send()
    3. Handles human-in-the-loop interruptions
    4. Coordinates state flow between agents
    """
    
    @log_sync
    def __init__(
        self,
        agents: List[BaseSubgraphAgent],
        config: Optional[Config] = None,
        custom_config: Optional[Dict[str, Any]] = None,
        prompt_template: Optional[str] = None,
        name: str = "supervisor-agent"
    ):
        """
        Initialize the supervisor agent with centralized configuration.
        
        Args:
            agents: List of subgraph agents to orchestrate
            config: Configuration instance (defaults to new Config())
            custom_config: Optional custom configuration to override defaults
            prompt_template: Custom prompt template for supervisor
            name: Agent name for identification
        """
        # Use centralized config system
        self.config_instance = config or Config(custom_config or {})
        
        # Set agent name for identification
        self._name = name
        
        # Get LLM configuration from centralized config
        llm_config = self.config_instance.get_llm_config()
        
        # Initialize the LLM model using the centralized provider
        try:
            self.model = LLMProvider.create_llm(
                provider=llm_config['provider'],
                model=llm_config['model'],
                temperature=llm_config['temperature'],
                max_tokens=llm_config['max_tokens']
            )
            supervisor_logger.log_structured(
                level="INFO",
                message=f"Initialized LLM model: {llm_config['provider']}:{llm_config['model']}",
                extra={"llm_provider": llm_config['provider'], "llm_model": llm_config['model']}
            )
        except Exception as e:
            supervisor_logger.log_structured(
                level="ERROR",
                message=f"Failed to initialize LLM model: {e}",
                extra={"error": str(e)}
            )
            raise
        
        # Get supervisor-specific configuration from centralized config
        self.supervisor_config = {
            "output_mode": self.config_instance.supervisor_output_mode,
            "add_handoff_back_messages": self.config_instance.supervisor_add_handoff_back_messages,
            "max_snapshots": self.config_instance.supervisor_max_snapshots,
            "enable_audit_trail": self.config_instance.supervisor_enable_audit_trail,
            "max_concurrent_workflows": self.config_instance.supervisor_max_concurrent_workflows,
            "workflow_timeout": self.config_instance.supervisor_workflow_timeout,
            "human_approval_required": self.config_instance.supervisor_human_approval_required,
            "validation_always_required": self.config_instance.supervisor_validation_always_required,
            "max_retries": self.config_instance.supervisor_max_retries,
            "timeout_seconds": self.config_instance.supervisor_timeout_seconds
        }
        
        # Initialize memory for human-in-the-loop first
        self.memory = MemorySaver()
        
        # Initialize agents and pass the checkpointer
        self.agents = {}
        for agent in agents:
            # Set the checkpointer for each agent
            if hasattr(agent, 'memory'):
                agent.memory = self.memory
            self.agents[agent.name] = agent
        
        # Set prompt template
        self.prompt_template = prompt_template or self._get_default_prompt()
        
        # Initialize supervisor state
        self.supervisor_state: Optional[SupervisorState] = None
        
        # Build the supervisor graph
        self.graph = self._build_supervisor_graph()
        self.compiled_graph = self.graph.compile(checkpointer=self.memory)
        
        # Validate that the graph was built correctly
        if not self.compiled_graph:
            raise ValueError("Failed to compile supervisor graph")
        
        supervisor_logger.log_structured(
            level="DEBUG",
            message="Supervisor graph compiled successfully",
            extra={
                "graph_nodes": list(self.graph.nodes.keys()) if hasattr(self.graph, 'nodes') else "unknown",
                "compiled_graph_type": type(self.compiled_graph).__name__
            }
        )
        
        supervisor_logger.log_structured(
            level="INFO",
            message="Custom Supervisor Agent initialized successfully with centralized config",
            extra={
                "agent_count": len(agents), 
                "name": name,
                "llm_provider": llm_config['provider'],
                "llm_model": llm_config['model'],
                "max_snapshots": self.supervisor_config["max_snapshots"]
            }
        )
    
    def _get_default_prompt(self) -> str:
        """Get the default prompt template for the supervisor following LangGraph tutorial pattern."""
        agent_names = list(self.agents.keys())
        agent_descriptions = "\n".join([f"- {name}: {self._get_agent_description(name)}" for name in agent_names])
        
        return f"""
        You are a supervisor managing specialized infrastructure agents for AWS Terraform module generation.
        
        Available agents:
        {agent_descriptions}
        
        Your responsibilities:
        1. Analyze the user request and determine which agent(s) to delegate to
        2. Route tasks to the appropriate agent using the available transfer_to_* tools
        3. Coordinate the workflow and handle any interruptions
        4. Ensure the final result meets the user's requirements
        
        WORKFLOW RULES:
        - For ANY infrastructure request (creating, writing, or generating Terraform modules), ALWAYS start by delegating to planner_sub_supervisor
        - The planner_sub_supervisor will analyze requirements and create execution plans
        - After planning is complete (when you receive planner_data), IMMEDIATELY delegate to generator_swarm to generate the actual Terraform code
        - After generation is complete (when you receive generation_data), IMMEDIATELY delegate to writer_react_agent to write the Terraform files to disk
        - Use validation_agent to validate the generated code if needed
        - Use editor_agent to modify existing configurations if requested
        
        CRITICAL: When planning is complete, you MUST continue the workflow by delegating to the next agent. Do not stop or wait.
        
        ROUTING DECISIONS:
        1. Check workflow_state.loop_counter - if > 20, terminate with error
        2. Check current workflow phase and completion status:
           - If workflow_state.current_phase == "planning" and not planning_complete:
             → transfer_to_planner_sub_supervisor
           - If workflow_state.current_phase == "planning" and planning_complete and not generation_complete:
             → transfer_to_generator_swarm
           - If workflow_state.current_phase == "generation" and generation_complete and not writing_complete:
             → transfer_to_writer_react_agent
           - If workflow_state.current_phase == "writing" and writing_complete and not validation_complete:
             → transfer_to_validation_agent
           - If workflow_state.current_phase == "validation" and validation_complete:
             → Workflow complete, return final result
        3. Check for specific user requests:
           - If user wants to modify existing code: → transfer_to_editor_agent
           - If user wants to validate existing code: → transfer_to_validation_agent
           - If user wants new infrastructure: → transfer_to_planner_sub_supervisor
        
        COMPLETION VALIDATION:
        - After each agent handoff, validate that the expected completion flag is set
        - Check workflow_state for phase completion flags:
          * planning_complete: True when planner_data is available and planning_complete=True
          * generation_complete: True when generation_data is available
          * writing_complete: True when writing_data is available
          * validation_complete: True when validation_data is available
        - If completion flag not set after reasonable time, log error and retry once
        - If retry fails, terminate workflow and escalate to human
        - **CRITICAL: When planning_complete == True, you MUST immediately delegate to generator_swarm**
        - **CRITICAL: When generation_complete == True, you MUST immediately delegate to writer_react_agent**
        - **DO NOT stop the workflow after planning or generation - ALWAYS continue to the next phase**
        
        Instructions:
        - Delegate one agent at a time, do not call agents in parallel
        - Use the transfer_to_* tools to delegate tasks
        - Provide clear task descriptions to agents
        - Handle any human-in-the-loop interruptions
        - Return the final result when all work is complete
        
        IMPORTANT: For infrastructure requests like "help me write aws sns terraform module", ALWAYS start with planner_sub_supervisor.
        
        Do not do any work yourself - only delegate to the appropriate agents.
        """
    
    def _get_agent_description(self, agent_name: str) -> str:
        """Get description for an agent based on its name."""
        descriptions = {
            "planner_sub_supervisor": "Analyzes requirements and creates execution plans using specialized sub-agents",
            "generator_swarm": "Generates Terraform modules using coordinated swarm of specialized generator agents (subgraph with isolated state)",
            "editor_agent": "Modifies existing Terraform configurations",
            "validation_agent": "Validates Terraform modules and configurations"
        }
        return descriptions.get(agent_name, "Specialized infrastructure agent")
    
    def _build_supervisor_graph(self) -> StateGraph:
        """Build the supervisor StateGraph using langgraph-supervisor."""
        
        # Get agent names for handoff tools (keep all agents including generator_swarm)
        agent_names = list(self.agents.keys())
        
        # Create custom handoff tools for all agents
        handoff_tools = create_handoff_tools_for_agents(agent_names)
        
        # Create agents list for create_supervisor
        # Each agent should be a compiled graph with a name
        agents = []
        for agent_name, agent in self.agents.items():
            if agent_name == "generator_swarm":
                # Special handling for generator_swarm - use transformation node wrapper
                supervisor_logger.log_structured(
                    level="INFO",
                    message="Creating generator_swarm transformation node wrapper",
                    extra={
                        "agent_name": agent_name,
                        "agent_type": type(agent).__name__,
                        "integration_type": "transformation_node_wrapper"
                    }
                )
                
                # Create transformation node wrapper function with proper closure
                # Capture the agent in the closure by creating a factory function
                def make_generator_swarm_transformation_node(generator_agent):
                    async def generator_swarm_transformation_node(supervisor_state: SupervisorState, config: dict = None, **kwargs) -> Dict[str, Any]:
                        """
                        Transformation node wrapper that handles state conversion between SupervisorState and GeneratorSwarmState.
                        This follows the LangGraph pattern for disjoint schemas with explicit state transformation.
                        
                        Args:
                            supervisor_state: SupervisorState from parent graph
                            config: Optional configuration dict (for langgraph-supervisor compatibility)
                            **kwargs: Additional keyword arguments (for langgraph-supervisor compatibility)
                        """
                        try:
                            # 1. Pass SupervisorState directly to wrapper (let wrapper handle transformation)
                            # This ensures tools receive the full GeneratorSwarmState via InjectedState
                            generator_output = await generator_agent.create_wrapper_function()(supervisor_state)
                            
                            return generator_output
                            
                        except Exception as e:
                            supervisor_logger.log_structured(
                                level="ERROR",
                                message="Generator swarm transformation node error",
                                extra={
                                    "error": str(e),
                                    "error_type": type(e).__name__
                                }
                            )
                            raise
                    return generator_swarm_transformation_node
                
                # Create the transformation node with proper agent capture
                generator_swarm_transformation_node = make_generator_swarm_transformation_node(agent)
                
                # Create wrapper object for langgraph-supervisor compatibility
                class GeneratorSwarmWrapper:
                    def __init__(self, wrapper_func, name):
                        self.name = name
                        self._wrapper_func = wrapper_func
                    
                    def __call__(self, *args, **kwargs):
                        return self._wrapper_func(*args, **kwargs)
                    
                    async def __acall__(self, *args, **kwargs):
                        return await self._wrapper_func(*args, **kwargs)
                    
                    async def ainvoke(self, *args, **kwargs):
                        return await self._wrapper_func(*args, **kwargs)
                    
                    def invoke(self, *args, **kwargs):
                        return self._wrapper_func(*args, **kwargs)
                
                # Create the wrapper object
                generator_swarm_wrapper = GeneratorSwarmWrapper(generator_swarm_transformation_node, "generator_swarm")
                agents.append(generator_swarm_wrapper)
                
                supervisor_logger.log_structured(
                    level="INFO",
                    message="Generator_swarm transformation node wrapper created successfully",
                    extra={
                        "agent_name": agent_name,
                        "wrapper_type": type(generator_swarm_wrapper).__name__
                    }
                )
                
            else:
                # Regular agent compilation
                supervisor_logger.log_structured(
                    level="DEBUG",
                    message=f"Compiling regular agent for supervisor",
                    extra={
                        "agent_name": agent_name,
                        "agent_type": type(agent).__name__,
                        "agent_has_build_graph": hasattr(agent, 'build_graph'),
                    }
                )
                
                supervisor_logger.log_structured(
                    level="DEBUG",
                    message=f"About to compile agent graph",
                    extra={
                        "agent_name": agent_name,
                        "agent_type": type(agent).__name__,
                        "agent_has_build_graph": hasattr(agent, 'build_graph'),
                        "agent_has_name": hasattr(agent, '_name'),
                        "agent_name_value": getattr(agent, '_name', 'unknown')
                    }
                )
                
                compiled_agent = agent.build_graph().compile(
                    # checkpointer=self.memory,
                    name=agent_name  # Set the agent name
                )
                
                supervisor_logger.log_structured(
                    level="DEBUG",
                    message=f"Agent compiled successfully",
                    extra={
                        "agent_name": agent_name,
                        "compiled_agent_type": type(compiled_agent).__name__,
                        "compiled_agent_has_nodes": hasattr(compiled_agent, 'nodes'),
                    }
                )
                
                agents.append(compiled_agent)
        
        # Create pre-model hook for workflow tracking and observability (SECONDARY METHOD)
        def pre_model_hook(state: SupervisorState) -> SupervisorState:
            """
            Pre-model hook for workflow tracking and observability.
            
            This hook provides:
            - Workflow progress tracking
            - Agent completion detection
            - Loop prevention
            - Debugging and monitoring
            """
            try:
                supervisor_logger.log_structured(
                    level="DEBUG",
                    message="Pre-model hook: Processing supervisor state",
                    extra={
                        "current_phase": state.workflow_state.current_phase,
                        "workflow_complete": state.workflow_state.workflow_complete,
                        "loop_counter": state.workflow_state.loop_counter,
                        "last_agent": state.workflow_state.last_agent,
                        "next_agent": state.workflow_state.next_agent
                    }
                )
                
                # Increment loop counter and check for errors
                state.workflow_state.increment_loop_counter()
                if state.workflow_state.error_occurred:
                    supervisor_logger.log_structured(
                        level="ERROR",
                        message="Loop limit exceeded in supervisor pre-model hook",
                        extra={
                            "loop_counter": state.workflow_state.loop_counter,
                            "error_message": state.workflow_state.error_message
                        }
                    )
                    return state
                
                # NEW: Parse planner data from message history if not already set
                if (not state.planner_data and state.messages):
                    
                    parsed_planner_data = self._parse_planner_data_from_tool_message(state.messages)
                    if parsed_planner_data:
                        state.planner_data = parsed_planner_data
                        
                        # Set current_agent to planner_sub_supervisor since we have planner data
                        state.current_agent = "planner"
                        
                        # Check if planning is complete and update workflow state accordingly
                        if (parsed_planner_data.get("planning_complete", False) and 
                            not state.workflow_state.planning_complete):
                            
                            # Mark planning phase as complete
                            state.workflow_state.set_phase_complete("planning")
                            
                            # Set up agent handoff to next phase
                            state.workflow_state.set_agent_handoff(
                                from_agent="planner_sub_supervisor",
                                to_agent="generator_swarm",
                                reason="Planning complete, proceeding to generation"
                            )
                            
                            supervisor_logger.log_structured(
                                level="INFO",
                                message="Successfully parsed planner data and updated workflow state - planning complete",
                                extra={
                                    "planner_data_keys": list(parsed_planner_data.keys()) if parsed_planner_data else [],
                                    "planning_complete": True,
                                    "current_agent": state.current_agent,
                                    "next_agent": "generator_swarm",
                                    "workflow_phase_updated": True
                                }
                            )
                        else:
                            supervisor_logger.log_structured(
                                level="INFO",
                                message="Successfully parsed planner data from message history",
                                extra={
                                    "planner_data_keys": list(parsed_planner_data.keys()) if parsed_planner_data else [],
                                    "planning_complete": parsed_planner_data.get("planning_complete", False),
                                    "current_agent": state.current_agent,
                                    "workflow_phase_updated": False
                                }
                            )
                    else:
                        supervisor_logger.log_structured(
                            level="DEBUG",
                            message="No planner data found in message history",
                            extra={"current_agent": state.current_agent}
                        )
                if (state.planner_data and state.generation_data and not state.writer_data):
                    writer_tool_message = None
                    for message in reversed(state.messages):
                        if (hasattr(message, 'name') and 
                            message.name == 'completion_tool' and
                            hasattr(message, 'content')):
                            writer_tool_message = message
                            break
                    
                    if not writer_tool_message:
                        supervisor_logger.log_structured(
                            level="DEBUG",
                            message="No writer_complete_task tool message found in message history",
                            extra={"total_messages": len(state.messages)}
                        )
                        return None
                    
                    # Parse the content as a dictionary
                    content = writer_tool_message.content
                    if not content or not isinstance(content, str):
                        supervisor_logger.log_structured(
                            level="WARNING",
                            message="Tool message content is empty or not a string",
                            extra={"content_type": type(content).__name__}
                        )
                        return None
                    
                    # Try to parse the content as a dictionary
                    # Since we now use model_dump(mode='json'), the content should be valid JSON
                    response = {}
                    if isinstance(content, str):
                        response = json.loads(content)
                    else:
                        response = content
                    completion_status = response.get("completion_status", "completed")
                    completion_summary = response.get("completion_summary", "")
                    completion_files_created = response.get("completion_files_created", [])

                    state.writer_data = {
                        "status": completion_status,
                        "summary": completion_summary,
                        "files_created": completion_files_created
                    }
                    state.workflow_state.set_phase_complete("writer")
                    supervisor_logger.log_structured(
                        level="INFO",
                        message="Successfully parsed writer data and updated workflow state - writer complete",
                        extra={
                            "writer_data_keys": list(state.writer_data.keys()) if state.writer_data else [],
                            "writer_complete": True,
                            "current_agent": state.current_agent,
                            "workflow_phase_updated": True
                        }
                    )
                # Process agent completions and update workflow state
                self._process_agent_completions(state)
                
                # Check if workflow should continue to next phase
                if self._should_continue_workflow(state):
                    self._prepare_next_phase(state)
                
                # CRITICAL: Add explicit completion context for LLM when planning is complete
                if (state.workflow_state.planning_complete and not state.workflow_state.generation_complete):
                    
                    completion_context = f"""PLANNING PHASE COMPLETE:
planning_complete = True
current_phase = {state.workflow_state.current_phase}
workflow_complete = {state.workflow_state.workflow_complete}

MANDATORY ACTION: You MUST delegate to generator_swarm to generate the Terraform code.
DO NOT delegate back to planner_sub_supervisor.
The planning workflow is complete and you must proceed to the generation phase."""
                    
                    # Set explicit completion context for LLM
                    state.llm_input_messages = [HumanMessage(content=completion_context)]
                    
                    supervisor_logger.log_structured(
                        level="INFO",
                        message="Added explicit completion context for LLM - planning complete",
                        extra={
                            "planning_complete": state.workflow_state.planning_complete,
                            "current_phase": state.workflow_state.current_phase,
                            "current_agent": state.current_agent
                        }
                    )
                
                elif (state.workflow_state.generation_complete and not state.workflow_state.writer_complete):
                    
                    completion_context = f"""GENERATION PHASE COMPLETE:
generation_complete = True
current_phase = {state.workflow_state.current_phase}
workflow_complete = {state.workflow_state.workflow_complete}

MANDATORY ACTION: You MUST delegate to writer_react_agent to write the Terraform code.
DO NOT delegate back to generator_swarm.
The generation workflow is complete and you must proceed to the writer phase.
"""

                    # Set explicit completion context for LLM
                    state.llm_input_messages = [HumanMessage(content=completion_context)]
                    
                    supervisor_logger.log_structured(
                        level="INFO",
                        message="Added explicit completion context for LLM - generation complete",
                        extra={
                            "generation_complete": state.workflow_state.generation_complete,
                            "current_phase": state.workflow_state.current_phase,
                            "current_agent": state.current_agent
                        }
                    )
                
                elif (state.workflow_state.writer_complete):
                    
                    completion_context = f"""WORKFLOW COMPLETE:
writer_complete = True
current_phase = {state.workflow_state.current_phase}
workflow_complete = {state.workflow_state.workflow_complete}

MANDATORY ACTION: ALL WORK IS COMPLETE. HURRAY!, Workflow complete. Do not delegate back to any agent.
"""

                    # Set explicit completion context for LLM
                    state.llm_input_messages = [HumanMessage(content=completion_context)]
                    
                    supervisor_logger.log_structured(
                        level="INFO",
                        message="Added explicit completion context for LLM - workflow complete",
                        extra={
                            "workflow_complete": state.workflow_state.workflow_complete,
                            "current_phase": state.workflow_state.current_phase,
                            "current_agent": state.current_agent
                        }
                    )

                return state
                
            except Exception as e:
                supervisor_logger.log_structured(
                    level="ERROR",
                    message=f"Error in pre-model hook: {e}",
                    extra={"error": str(e), "error_type": type(e).__name__}
                )
                return state
        
        # Create supervisor using langgraph-supervisor with our custom state schema
        supervisor_graph = create_supervisor(
            agents=agents,
            model=self.model,
            tools=handoff_tools,
            prompt=self.prompt_template,
            state_schema=SupervisorState,  # Use our custom state schema
            add_handoff_back_messages=True,
            output_mode="full_history",
            pre_model_hook=pre_model_hook  # Add pre-model hook for workflow tracking
        )
        
        # Debug: Log the supervisor graph structure
        supervisor_logger.log_structured(
            level="DEBUG",
            message="Supervisor graph created with langgraph-supervisor",
            extra={
                "supervisor_graph_type": type(supervisor_graph).__name__,
                "supervisor_graph_has_nodes": hasattr(supervisor_graph, 'nodes'),
                "supervisor_graph_nodes": list(supervisor_graph.nodes.keys()) if hasattr(supervisor_graph, 'nodes') else "unknown",
                "handoff_tools_count": len(handoff_tools),
                "handoff_tool_names": [tool.name for tool in handoff_tools],
            }
        )
        
        supervisor_logger.log_structured(
            level="INFO",
            message="Created supervisor using langgraph-supervisor",
            extra={
                "agent_count": len(agents),
                "handoff_tools_count": len(handoff_tools),
                "agent_names": agent_names,
            }
        )
        
        return supervisor_graph
    
    
    def _transform_state_for_agent(self, agent_name: str, supervisor_state: SupervisorState, task_description: str) -> Dict[str, Any]:
        """
        Transform supervisor state to agent-specific state.
        
        This method ensures that each agent receives only the data it needs
        and doesn't interfere with other agents' state schemas.
        
        Args:
            agent_name: Name of the target agent
            supervisor_state: Current supervisor state
            task_description: Task description for the agent
            
        Returns:
            Agent-specific state as dictionary
        """
        try:
            # Get the appropriate state class for this agent
            agent_type = self._get_agent_type_from_name(agent_name)
            
            # Create agent-specific state without workflow_state to avoid conflicts
            agent_state = {
                "user_request": task_description,
                "session_id": supervisor_state.session_id,
                "task_id": supervisor_state.task_id,
                "status": supervisor_state.status.value if hasattr(supervisor_state.status, 'value') else str(supervisor_state.status),
                "workflow_id": supervisor_state.workflow_id,
            }
            
            # Add context if available
            if supervisor_state.terraform_context:
                agent_state["context"] = supervisor_state.terraform_context
            
            # IMPORTANT: Do NOT pass supervisor's workflow_state to other agents
            # Each agent should manage its own workflow state independently
            # This prevents schema conflicts between different workflow state types
            
            supervisor_logger.log_structured(
                level="DEBUG",
                message=f"Transformed state for {agent_name} - excluded workflow_state to prevent schema conflicts",
                extra={
                    "agent_name": agent_name,
                    "agent_type": agent_type.value,
                    "state_keys": list(agent_state.keys()),
                    "session_id": agent_state.get("session_id"),
                    "task_id": agent_state.get("task_id"),
                    "note": "workflow_state excluded to prevent PlannerSupervisorState validation errors"
                }
            )
            
            return agent_state
            
        except Exception as e:
            supervisor_logger.log_structured(
                level="ERROR",
                message=f"Failed to transform state for {agent_name}: {e}",
                extra={
                    "agent_name": agent_name,
                    "error": str(e),
                    "workflow_id": supervisor_state.workflow_id
                }
            )
            # Fallback to basic state
            return {
                "user_request": task_description,
                "session_id": supervisor_state.session_id,
                "task_id": supervisor_state.task_id,
            }
    
    def _get_agent_type_from_name(self, agent_name: str) -> AgentType:
        """Get agent type from agent name."""
        name_to_type = {
            "planner_sub_supervisor": AgentType.PLANNER,
            "generator_swarm": AgentType.GENERATION,  # Generator swarm is also a generation agent
            "generation_agent": AgentType.GENERATION,
            "validation_agent": AgentType.VALIDATION,
            "editor_agent": AgentType.EDITOR,
            "security_agent": AgentType.SECURITY,
            "cost_agent": AgentType.COST
        }
        return name_to_type.get(agent_name, AgentType.PLANNER)
    
    def _process_agent_completions(self, state: SupervisorState) -> None:
        """
        Process agent completions and update workflow state.
        
        This method detects when agents complete their work and updates
        the workflow state accordingly, following best practices for
        state management.
        """
        try:
            # Check for planner completion
            if (state.planner_data and 
                not state.workflow_state.planning_complete and
                self._is_planner_complete(state.planner_data)):
                
                supervisor_logger.log_structured(
                    level="INFO",
                    message="Planning phase completed - updating workflow state",
                    extra={
                        "current_phase": state.workflow_state.current_phase,
                        "planning_complete": True
                    }
                )
                
                state.workflow_state.set_phase_complete("planning")
                state.workflow_state.set_agent_handoff(
                    from_agent="planner_sub_supervisor",
                    to_agent="generator_swarm",
                    reason="Planning complete, proceeding to generation"
                )
            
            # Check for generation completion
            if (state.generation_data and 
                not state.workflow_state.generation_complete and
                self._is_generation_complete(state.generation_data)):
                
                supervisor_logger.log_structured(
                    level="INFO",
                    message="Generation phase completed - updating workflow state",
                    extra={
                        "current_phase": state.workflow_state.current_phase,
                        "generation_complete": True
                    }
                )
                
                state.workflow_state.set_phase_complete("generation")
                state.workflow_state.set_agent_handoff(
                    from_agent="generator_swarm",
                    to_agent="writer_react_agent",  # Handoff to writer react agent
                    reason="Generation complete, handoff to writer react agent"
                )
            
            # Check for validation completion (optional phase)
            if (state.writer_data and 
                not state.workflow_state.writer_complete and
                self._is_writer_complete(state.writer_data)):
                
                supervisor_logger.log_structured(
                    level="INFO",
                    message="Writer phase completed - updating workflow state",
                    extra={
                        "current_phase": state.workflow_state.current_phase,
                        "writer_complete": True
                    }
                )
                
                state.workflow_state.set_phase_complete("writer")
            
            # Check for editing completion (optional phase)
            if (state.editor_data and 
                not state.workflow_state.editing_complete and
                self._is_editing_complete(state.editor_data)):
                
                supervisor_logger.log_structured(
                    level="INFO",
                    message="Editing phase completed - updating workflow state",
                    extra={
                        "current_phase": state.workflow_state.current_phase,
                        "editing_complete": True
                    }
                )
                
                state.workflow_state.set_phase_complete("editing")
                
        except Exception as e:
            supervisor_logger.log_structured(
                level="ERROR",
                message=f"Error processing agent completions: {e}",
                extra={"error": str(e), "error_type": type(e).__name__}
            )
    
    def _is_planner_complete(self, planner_data) -> bool:
        """Check if planner phase is complete."""
        if not planner_data:
            return False
        
        # Use the explicit completion flag from planner
        # Handle both dict and PlannerData object
        if hasattr(planner_data, 'planning_complete'):
            return planner_data.planning_complete
        else:
            return planner_data.get("planning_complete", False)
    
    def _is_generation_complete(self, generation_data: Dict[str, Any]) -> bool:
        """Check if generation phase is complete."""
        if not generation_data:
            return False
        
        # Check for generated module or completion status
        return (generation_data.get("generated_module") is not None or
                generation_data.get("status") == "completed")
    
    def _is_writer_complete(self, writer_data: Dict[str, Any]) -> bool:
        """Check if validation phase is complete."""
        if not writer_data:
            return False
        
        # Check for module reference or completion status
        return (writer_data.get("module_ref") is not None or
                writer_data.get("status") == "completed")
    
    def _is_editing_complete(self, editor_data: Dict[str, Any]) -> bool:
        """Check if editing phase is complete."""
        if not editor_data:
            return False
        
        # Check for edited module or completion status
        return (editor_data.get("edited_module") is not None or
                editor_data.get("status") == "completed")
    
    def _parse_planner_data_from_tool_message(self, messages: List[AnyMessage]) -> Optional[Dict[str, Any]]:
        """
        Parse planner data from the mark_planning_complete tool message.
        
        Looks for the last ToolMessage with name='mark_planning_complete' and
        parses its content as planner data.
        
        Args:
            messages: List of messages from the state
            
        Returns:
            Parsed planner data dictionary or None if not found/invalid
        """
        try:
            # Find the last ToolMessage with name='mark_planning_complete'
            planner_tool_message = None
            for message in reversed(messages):
                if (hasattr(message, 'name') and 
                    message.name == 'mark_planning_complete' and
                    hasattr(message, 'content')):
                    planner_tool_message = message
                    break
            
            if not planner_tool_message:
                supervisor_logger.log_structured(
                    level="DEBUG",
                    message="No mark_planning_complete tool message found in message history",
                    extra={"total_messages": len(messages)}
                )
                return None
            
            # Parse the content as a dictionary
            content = planner_tool_message.content
            if not content or not isinstance(content, str):
                supervisor_logger.log_structured(
                    level="WARNING",
                    message="Tool message content is empty or not a string",
                    extra={"content_type": type(content).__name__}
                )
                return None
            
            # Try to parse the content as a dictionary
            # Since we now use model_dump(mode='json'), the content should be valid JSON
            try:
                parsed_data = json.loads(content)
                if isinstance(parsed_data, dict):
                    supervisor_logger.log_structured(
                        level="INFO",
                        message="Successfully parsed planner data from tool message using JSON",
                        extra={
                            "data_keys": list(parsed_data.keys()) if parsed_data else [],
                            "has_requirements_data": bool(parsed_data.get("requirements_data")),
                            "has_execution_data": bool(parsed_data.get("execution_data")),
                            "has_planning_results": bool(parsed_data.get("planning_results")),
                            "planning_complete": parsed_data.get("planning_complete", False)
                        }
                    )
                    return parsed_data
                else:
                    supervisor_logger.log_structured(
                        level="WARNING",
                        message="JSON parsed content is not a dictionary",
                        extra={"parsed_type": type(parsed_data).__name__}
                    )
                    return None
                    
            except (json.JSONDecodeError, TypeError) as json_error:
                # If JSON parsing fails, try ast.literal_eval as fallback for simple cases
                try:
                    parsed_data = ast.literal_eval(content)
                    if isinstance(parsed_data, dict):
                        supervisor_logger.log_structured(
                            level="INFO",
                            message="Successfully parsed planner data from tool message using ast.literal_eval fallback",
                            extra={
                                "data_keys": list(parsed_data.keys()) if parsed_data else [],
                                "has_requirements_data": bool(parsed_data.get("requirements_data")),
                                "has_execution_data": bool(parsed_data.get("execution_data")),
                                "has_planning_results": bool(parsed_data.get("planning_results")),
                                "planning_complete": parsed_data.get("planning_complete", False)
                            }
                        )
                        return parsed_data
                    else:
                        supervisor_logger.log_structured(
                            level="WARNING",
                            message="ast.literal_eval parsed content is not a dictionary",
                            extra={"parsed_type": type(parsed_data).__name__}
                        )
                        return None
                        
                except (ValueError, SyntaxError) as ast_error:
                    supervisor_logger.log_structured(
                        level="ERROR",
                        message="Failed to parse planner data from tool message content",
                        extra={
                            "content_preview": content[:200] + "..." if len(content) > 200 else content,
                            "json_error": str(json_error),
                            "ast_error": str(ast_error)
                        }
                    )
                    return None
                    
        except Exception as e:
            supervisor_logger.log_structured(
                level="ERROR",
                message="Unexpected error parsing planner data from tool message",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            return None
    
    def _should_continue_workflow(self, state: SupervisorState) -> bool:
        """Determine if workflow should continue to next phase."""
        # Don't continue if workflow is already complete
        if state.workflow_state.workflow_complete:
            return False
        
        # Don't continue if there's an error
        if state.workflow_state.error_occurred:
            return False
        
        # Continue if there's a next phase available
        return state.workflow_state.next_phase is not None
    
    def _prepare_next_phase(self, state: SupervisorState) -> None:
        """Prepare the next phase of the workflow."""
        try:
            next_phase = state.workflow_state.next_phase
            if not next_phase:
                return
            
            supervisor_logger.log_structured(
                level="INFO",
                message=f"Preparing next phase: {next_phase}",
                extra={
                    "current_phase": state.workflow_state.current_phase,
                    "next_phase": next_phase,
                    "workflow_progress": state.workflow_state.get_workflow_progress()
                }
            )
            
            # Update status to indicate workflow should continue
            state.status = WorkflowStatus.IN_PROGRESS
            
            # Set current agent based on next phase
            if next_phase == "planning":
                state.current_agent = AgentType.PLANNER
            elif next_phase == "generation":
                state.current_agent = AgentType.GENERATION
            elif next_phase == "writer":
                state.current_agent = AgentType.WRITER
            elif next_phase == "editing":
                state.current_agent = AgentType.EDITOR
            
        except Exception as e:
            supervisor_logger.log_structured(
                level="ERROR",
                message=f"Error preparing next phase: {e}",
                extra={"error": str(e), "error_type": type(e).__name__}
            )
    
    def _merge_agent_state_back_to_supervisor(self, agent_name: str, agent_state: Dict[str, Any], supervisor_state: SupervisorState) -> SupervisorState:
        """
        Merge agent state back into supervisor state.
        
        This method handles the state propagation back from subgraph agents to supervisor.
        For the planner sub-supervisor, it directly uses the agent's output_transform() result.
        
        Args:
            agent_name: Name of the agent that returned
            agent_state: State returned from the agent
            supervisor_state: Current supervisor state
            
        Returns:
            Updated supervisor state with agent results merged
        """
        try:
            agent_type = self._get_agent_type_from_name(agent_name)
            
            # Handle planner sub-supervisor specially (it manages its own state)
            if agent_type == AgentType.PLANNER:
                # The planner handoff tool passes planner_data directly
                # Extract planner_data from the agent state
                planner_data = agent_state.get("planner_data", {})
                
                # Handle both dict and PlannerData object
                if hasattr(planner_data, 'model_dump'):
                    # It's a PlannerData object, convert to dict for compatibility
                    planner_data_dict = planner_data.model_dump()
                else:
                    # It's already a dict
                    planner_data_dict = planner_data
                
                supervisor_updates = {
                    "messages": agent_state.get("messages", []),
                    "planner_data": planner_data_dict,  # Use dict version for compatibility
                    "status": WorkflowStatus.IN_PROGRESS,  # Keep workflow in progress to continue to next agent
                    "current_agent": None,  # Planning is complete, supervisor will determine next agent
                }
                
                # Update workflow state using the new tracking system
                # Only update supervisor's workflow state, not the agent's
                supervisor_state.workflow_state.set_phase_complete("planning")
                supervisor_state.workflow_state.set_agent_handoff(
                    from_agent="planner_sub_supervisor",
                    to_agent="generator_swarm",
                    reason="Planning complete, proceeding to generation"
                )
                
                # Extract execution plan data for direct access
                if planner_data and "execution_data" in planner_data:
                    execution_data = planner_data["execution_data"]
                    if execution_data and "execution_plan_data" in execution_data:
                        execution_plan_data = execution_data["execution_plan_data"]
                        if execution_plan_data and "execution_plans" in execution_plan_data:
                            execution_plans = execution_plan_data["execution_plans"]
                            if execution_plans and len(execution_plans) > 0:
                                # Extract the first execution plan for direct access
                                first_plan = execution_plans[0]
                                supervisor_updates.update({
                                    "execution_plan": first_plan,
                                    "resource_configurations": first_plan.get("resource_configurations", []),
                                    "variable_definitions": first_plan.get("variable_definitions", []),
                                    "data_sources": first_plan.get("data_sources", []),
                                    "local_values": first_plan.get("local_values", []),
                                    "module_name": first_plan.get("module_name", "terraform-module"),
                                    "service_name": first_plan.get("service_name", "Unknown Service"),
                                    "target_environment": first_plan.get("target_environment", "prod")
                                })
                
                # Handle any questions from the planner
                if "agent_metadata" in agent_state:
                    metadata = agent_state["agent_metadata"]
                    if metadata.get("question"):
                        supervisor_updates["question"] = metadata["question"]
            else:
                # Use StateTransformer for other agents
                if agent_type == AgentType.GENERATION:
                    # Handle generator swarm specially (it has its own output_transform method)
                    if agent_name == "generator_swarm":
                        # The generator swarm has its own output_transform method
                        # We need to call it to get the proper state transformation
                        generator_agent = self.agents.get("generator_swarm")
                        if generator_agent and hasattr(generator_agent, 'output_transform'):
                            supervisor_updates = generator_agent.output_transform(agent_state)
                        else:
                            # Fallback to StateTransformer
                            supervisor_updates = StateTransformer.generator_swarm_to_supervisor(GeneratorSwarmState(**agent_state))
                    else:
                        # Regular generation agent
                        supervisor_updates = StateTransformer.generation_to_supervisor(GenerationState(**agent_state))
                elif agent_type == AgentType.VALIDATION:
                    supervisor_updates = StateTransformer.validation_to_supervisor(ValidationState(**agent_state))
                elif agent_type == AgentType.EDITOR:
                    supervisor_updates = StateTransformer.editor_to_supervisor(EditorState(**agent_state))
                elif agent_type == AgentType.SECURITY:
                    supervisor_updates = StateTransformer.security_to_supervisor(SecurityState(**agent_state))
                elif agent_type == AgentType.COST:
                    supervisor_updates = StateTransformer.cost_to_supervisor(CostState(**agent_state))
                else:
                    # Fallback for unknown agent types
                    supervisor_updates = {}
            
            # Merge messages (LangGraph automatically handles this, but we ensure it's done properly)
            if "messages" in agent_state:
                supervisor_updates["messages"] = agent_state["messages"]
            
            # Create updated supervisor state
            updated_state_data = supervisor_state.model_dump()
            updated_state_data.update(supervisor_updates)
            
            # Handle workflow completion
            if supervisor_updates.get("current_agent") is None:
                updated_state_data.update({
                    "status": WorkflowStatus.COMPLETED,
                    "workflow_completed_at": datetime.now(timezone.utc)
                })
            
            # Handle any errors from agents
            if "error" in agent_state:
                updated_state_data.update({
                    "error": agent_state["error"],
                    "status": WorkflowStatus.FAILED,
                    "current_agent": None
                })
                supervisor_logger.log_structured(
                    level="ERROR",
                    message=f"Agent {agent_name} returned error",
                    extra={
                        "agent_name": agent_name,
                        "error": agent_state["error"],
                        "status": WorkflowStatus.FAILED.value
                    }
                )
            
            # Create and validate the updated supervisor state
            updated_supervisor_state = SupervisorState(**updated_state_data)
            
            supervisor_logger.log_structured(
                level="DEBUG",
                message=f"Successfully merged {agent_name} state back to supervisor",
                extra={
                    "agent_name": agent_name,
                    "agent_type": agent_type.value,
                    "status": updated_supervisor_state.status.value,
                    "next_agent": supervisor_updates.get("current_agent")
                }
            )
            
            return updated_supervisor_state
            
        except Exception as e:
            supervisor_logger.log_structured(
                level="ERROR",
                message=f"Failed to merge {agent_name} state: {e}",
                extra={
                    "agent_name": agent_name,
                    "error": str(e),
                    "agent_state_keys": list(agent_state.keys()) if isinstance(agent_state, dict) else "not_dict"
                }
            )
            # Return original state if merge fails
            return supervisor_state
    
    @property
    def name(self) -> str:
        """Get the name of the supervisor agent."""
        return self._name

    @log_async
    async def stream(
        self,
        query_or_command,
        context_id: str,
        task_id: str
    ) -> AsyncGenerator[AgentResponse, None]:
        """
        Simplified async stream method following the reference pattern.
        
        Features:
        - Simple state management using StateTransformer
        - Clean interrupt detection with '__interrupt__' key
        - Direct status-based response formatting
        - Minimal manual state handling
        """
        supervisor_logger.log_structured(
            level="INFO",
            message=f"[stream] START",
            task_id=task_id,
            context_id=context_id,
            extra={
                "agent_name": self.__class__.__name__, 
                "query_or_command": str(query_or_command),
                "is_resume": isinstance(query_or_command, Command)
            }
        )

        # Simple init/resume handling
        if isinstance(query_or_command, Command):
            # Resume call: extract resume value and inject into state
            resume_value = query_or_command.resume
            graph_input = {
                "messages": [],
                "session_id": context_id,
                "task_id": task_id,
                "user_request": "",
                "status": "resuming",
                "resume_value": resume_value
            }
            thread_id = context_id  # Reuse for HITL resume
        else:
            # Initial call: build input state with new thread_id
            user_query = str(query_or_command)
            
            # Create the initial message with session_id and task_id in metadata
            initial_message = HumanMessage(
                content=user_query,
                additional_kwargs={
                    "session_id": context_id,
                    "task_id": task_id,
                    "user_request": user_query
                }
            )
            
            # For langgraph-supervisor, we need to use a simpler state structure
            graph_input = {
                "messages": [initial_message],
                "session_id": context_id,
                "task_id": task_id,
                "user_request": user_query,
                "status": "pending"
            }
            thread_id = str(uuid.uuid4())  # Unique per new user-initiated task

        config: RunnableConfig = {'configurable': {'thread_id': thread_id}}
        step_count = 0
        
        # Debug: Log the initial graph input state
        supervisor_logger.log_structured(
            level="DEBUG",
            message=f"Initial graph input state",
            task_id=task_id,
            context_id=context_id,
            extra={
                "graph_input_type": type(graph_input).__name__,
                "session_id": graph_input.get("session_id") if isinstance(graph_input, dict) else getattr(graph_input, "session_id", None),
                "task_id": graph_input.get("task_id") if isinstance(graph_input, dict) else getattr(graph_input, "task_id", None),
                "user_request": graph_input.get("user_request") if isinstance(graph_input, dict) else getattr(graph_input, "user_request", None),
                "messages_count": len(graph_input.get("messages", [])) if isinstance(graph_input, dict) else len(getattr(graph_input, "messages", [])),
            }
        )
        config_with_durability = {
            **config,
            "durability": "async",
            "subgraphs": True
        }
        
        try:
            # Use astream with values mode and subgraphs=True for proper handoff processing
            # This ensures proper subgraph handoff processing and prevents NoneType iteration errors
            async with isolation_shield():
                async for item in self.compiled_graph.astream(graph_input, config_with_durability, stream_mode='values', subgraphs=True):
                    step_count += 1
                    
                    # Handle None items from langgraph-supervisor
                    if item is None:
                        supervisor_logger.log_structured(
                            level="WARNING",
                            message=f"[stream] step={step_count} - Received None item from langgraph-supervisor",
                            task_id=task_id,
                            context_id=context_id,
                            extra={
                                "agent_name": self.__class__.__name__,
                                "step_count": step_count,
                                "item_type": "None"
                            }
                        )
                        continue
                
                    supervisor_logger.log_structured(
                        level="DEBUG",
                        message=f"[stream] step={step_count}",
                        task_id=task_id,
                        context_id=context_id,
                        extra={
                            "agent_name": self.__class__.__name__,
                            "step_count": step_count,
                            "item_keys": list(item.keys()) if isinstance(item, dict) else "not_dict",
                            "item_type": type(item).__name__
                        }
                    )

                    # Handle tuple format from subgraphs=True
                    if isinstance(item, tuple) and len(item) == 2:
                        # Extract state update from tuple (node_id, state_update)
                        node_id, state_update = item
                        item = state_update  # Use the state update part
                        supervisor_logger.log_structured(
                            level="DEBUG",
                            message=f"[stream] step={step_count} - Extracted state from tuple",
                            task_id=task_id,
                            context_id=context_id,
                            extra={
                                "agent_name": self.__class__.__name__,
                                "step_count": step_count,
                                "node_id": str(node_id),
                                "state_type": type(state_update).__name__,
                                "state_keys": list(state_update.keys()) if isinstance(state_update, dict) else "not_dict",
                                "status": state_update.get("status") if isinstance(state_update, dict) else None,
                                "active_agent": state_update.get("active_agent") if isinstance(state_update, dict) else None,
                                "planning_complete": state_update.get("planning_complete") if isinstance(state_update, dict) else None,
                                "has_planner_data": bool(state_update.get("planner_data")) if isinstance(state_update, dict) else False,
                                "planner_data_keys": list(state_update.get("planner_data", {}).keys()) if isinstance(state_update, dict) and state_update.get("planner_data") else [],
                                "has_requirements_data": bool(state_update.get("planner_data", {}).get("requirements_data")) if isinstance(state_update, dict) else False,
                                "has_execution_data": bool(state_update.get("planner_data", {}).get("execution_data")) if isinstance(state_update, dict) else False,
                                "has_planning_results": bool(state_update.get("planner_data", {}).get("planning_results")) if isinstance(state_update, dict) else False,
                                "messages_count": len(state_update.get("messages", [])) if isinstance(state_update, dict) else 0,
                                "task_description": str(state_update.get("task_description", ""))[:100] if isinstance(state_update, dict) else ""
                            }
                        )
                    elif not isinstance(item, dict):
                        supervisor_logger.log_structured(
                            level="WARNING",
                            message=f"[stream] step={step_count} - Received non-dict item from langgraph-supervisor",
                            task_id=task_id,
                            context_id=context_id,
                            extra={
                                "agent_name": self.__class__.__name__,
                                "step_count": step_count,
                                "item_type": type(item).__name__,
                                "item_value": str(item)[:200] if item else "None"
                            }
                        )
                        continue
                    # 1. Handle human-in-the-loop interrupt (simple check like reference)
                    if '__interrupt__' in item:
                        interrupt_payload = item['__interrupt__'][0].value  # dict passed to interrupt()
                        yield AgentResponse(
                            response_type='human_input',
                            is_task_complete=False,
                            require_user_input=True,
                            content=interrupt_payload.get('question', 'Input required'),
                            metadata={
                                'session_id': context_id,
                                'task_id': task_id,
                                'agent_name': self.name,
                                'step_count': step_count,
                                'status': 'input_required'
                            }
                        )
                        # Pause streaming until client resumes with feedback
                        break
                    # 2. Handle normal state updates (direct status-based responses like reference)
                    status = item.get('status')
                    question = item.get('question')
                    error = item.get('error')

                    if status is not None:
                        if status == 'input_required':
                            yield AgentResponse(
                                response_type='text',
                                is_task_complete=False,
                                require_user_input=True,
                                content=item.get('question') or 'More information needed to proceed.',
                                metadata={
                                    'session_id': context_id,
                                    'task_id': task_id,
                                    'agent_name': self.name,
                                    'step_count': step_count,
                                    'status': 'input_required'
                                }
                            )
                        elif status == 'error' or status == 'failed':
                            yield AgentResponse(
                                response_type='text',
                                is_task_complete=False,
                                require_user_input=True,
                                content=item.get('question') or 'An error occurred while processing your request.',
                                metadata={
                                    'session_id': context_id,
                                    'task_id': task_id,
                                    'agent_name': self.name,
                                    'step_count': step_count,
                                    'status': 'failed'
                                }
                            )
                        elif status == 'completed':
                            # Get planning data and status for individual agent completion
                            is_complete, planning_status = self._validate_planner_agent_completion(item)
                            
                            if is_complete:
                                # Extract the actual planning data
                                planning_data = planning_status.get('data_values', {})
                                completion_metrics = planning_status.get('completion_metrics', {})
                                
                                # Create comprehensive content with real planning data
                                content_data = {
                                    'status': 'planning_complete',
                                    'completion_metrics': completion_metrics,
                                    'requirements_data': planning_data.get('requirements_data', ''),
                                    'execution_data': planning_data.get('execution_data', ''),
                                    'planning_state': planning_status.get('planning_state', {}),
                                    'validation_result': planning_status.get('validation_result', 'PASS')
                                }
                                
                                yield AgentResponse(
                                    response_type='text',
                                    is_task_complete=False,
                                    require_user_input=False,
                                    content=str(content_data),
                                    metadata={
                                        'session_id': context_id,
                                        'task_id': task_id,
                                        'agent_name': self.name,
                                        'step_count': step_count,
                                        'status': 'planning_data_available',
                                        'planning_complete': True
                                    }
                                )
                                continue  # Continue to allow handoff tool execution
                            else:
                                # Individual agent completed, but supervisor workflow not fully complete
                                yield AgentResponse(
                                    response_type='text',
                                    is_task_complete=False,
                                    require_user_input=False,
                                    content=f'Agent completed - continuing supervisor workflow...',
                                    metadata={
                                        'session_id': context_id,
                                        'task_id': task_id,
                                        'agent_name': self.name,
                                        'step_count': step_count,
                                        'status': 'agent_completed_workflow_continuing'
                                    }
                                )
                                continue
                        else:
                            yield AgentResponse(
                                response_type='text',
                                is_task_complete=False,
                                require_user_input=False,
                                content=f'Processing... Status: {status}',
                                metadata={
                                    'session_id': context_id,
                                    'task_id': task_id,
                                    'agent_name': self.name,
                                    'step_count': step_count,
                                    'status': 'working'
                                }
                            )
                    else:
                        # Default processing response
                        yield AgentResponse(
                            response_type='text',
                            is_task_complete=False,
                            require_user_input=False,
                            content='Processing...',
                            metadata={
                                'session_id': context_id,
                                'task_id': task_id,
                                'agent_name': self.name,
                                'step_count': step_count,
                                'status': 'working'
                            }
                        )
                    
        except TypeError as e:
            # Handle specific NoneType iteration error from langgraph-supervisor handoff
            if "'NoneType' object is not iterable" in str(e):
                supervisor_logger.log_structured(
                    level="INFO",
                    message="Normal stream termination after handoff - NoneType iteration detected",
                    task_id=task_id,
                    context_id=context_id,
                    extra={
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "thread_id": thread_id,
                        "step_count": step_count,
                        "note": "This is expected behavior when subgraph handoff completes"
                    }
                )
                # Return a completion response instead of error
                yield AgentResponse(
                    response_type='data',
                    is_task_complete=True,
                    require_user_input=False,
                    content={'status': 'completed', 'message': 'Workflow completed successfully'},
                    metadata={
                        'session_id': context_id,
                        'task_id': task_id,
                        'agent_name': self.name,
                        'step_count': step_count,
                        'status': 'completed'
                    }
                )
                return
            else:
                # Re-raise other TypeError exceptions
                raise
        except Exception as e:
            supervisor_logger.log_structured(
                level="ERROR",
                message=f"Stream execution failed: {e}",
                task_id=task_id,
                context_id=context_id,
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "thread_id": thread_id,
                    "step_count": step_count
                }
            )
            yield AgentResponse(
                response_type='error',
                is_task_complete=True,
                require_user_input=False,
                content=f'Error during streaming: {str(e)}',
                error=str(e),
                metadata={
                    'session_id': context_id,
                    'task_id': task_id,
                    'agent_name': self.name,
                    'step_count': step_count,
                    'status': 'error'
                }
            )
        
            supervisor_logger.log_structured(
                level="INFO",
                message=f"[stream] END",
                task_id=task_id,
                context_id=context_id,
                extra={
                    "agent_name": self.__class__.__name__,
                    "thread_id": thread_id,
                    "step_count": step_count
                }
            )
    
    
    def _is_agent_return(self, item: Dict[str, Any]) -> bool:
        """Check if item represents a return from a subgraph agent."""
        # Check for agent-specific data fields that indicate completion
        agent_data_fields = [
            "planner_data", "generation_data", "validation_data", 
            "editor_data", "security_data", "cost_data"
        ]
        
        return any(field in item for field in agent_data_fields)
    
    def _extract_agent_name_from_return(self, item: Dict[str, Any]) -> Optional[str]:
        """Extract agent name from return item."""
        # Check for agent-specific data fields
        agent_data_map = {
            "planner_data": "planner_sub_supervisor",
            "generation_data": "generator_swarm",  # Generator swarm is the primary generation agent
            "validation_data": "validation_agent",
            "editor_data": "editor_agent",
            "security_data": "security_agent",
            "cost_data": "cost_agent"
        }
        
        for field, agent_name in agent_data_map.items():
            if field in item:
                return agent_name
        
        return None

    def _has_interrupt_data(self, item: Dict[str, Any]) -> bool:
        """Check if item contains interrupt data."""
        return isinstance(item, dict) and 'interrupt_data' in item

    def _extract_interrupt_data(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Extract interrupt data from item."""
        return item.get('interrupt_data', {})
    
    def _has_interrupt_context(self, item: Dict[str, Any]) -> bool:
        """Check if item contains interrupt context from agent."""
        return isinstance(item, dict) and (
            'interrupt_required' in item or 
            'interrupt_context' in item or
            item.get('interrupt_required', False)
        )
    
    def _extract_interrupt_context(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Extract interrupt context from agent return item."""
        if 'interrupt_context' in item:
            return item['interrupt_context']
        elif 'interrupt_required' in item and item['interrupt_required']:
            # Fallback: create basic interrupt data
            return {
                "context": "agent_interrupt",
                "question": "Agent requires human input",
                "agent_data": item
            }
        else:
            return {}
    
    @log_sync
    def _extract_graph_interrupt_data(self, graph_interrupt: Exception) -> Dict[str, Any]:
        """
        Extract interrupt data from GraphInterrupt exception.
        
        Args:
            graph_interrupt: The GraphInterrupt exception
            
        Returns:
            Extracted interrupt data
        """
        try:
            # GraphInterrupt typically contains the interrupt data in its args
            if hasattr(graph_interrupt, 'args') and graph_interrupt.args:
                # The first argument is usually the interrupt data
                interrupt_arg = graph_interrupt.args[0]
                if hasattr(interrupt_arg, 'value'):
                    return interrupt_arg.value
                elif isinstance(interrupt_arg, dict):
                    return interrupt_arg
                else:
                    return {"message": str(interrupt_arg)}
            
            # Fallback to string representation
            return {"message": str(graph_interrupt)}
            
        except Exception as e:
            supervisor_logger.log_structured(
                level="WARNING",
                message="Failed to extract interrupt data from GraphInterrupt",
                extra={"error": str(e), "graph_interrupt": str(graph_interrupt)}
            )
            return {"message": "Interrupt occurred", "error": str(graph_interrupt)}
    
    # ============================================================================
    # SIMPLIFIED UTILITY METHODS
    # ============================================================================
    
    @log_sync
    def _initialize_workflow(self, user_request: str, context_id: str, task_id: str) -> None:
        """Initialize a new workflow with the given user request."""
        self.supervisor_state = SupervisorState(
            user_request=user_request,
            session_id=context_id,
            task_id=task_id,
            status="pending",
            workflow_started_at=datetime.now()
        )
        
        supervisor_logger.log_structured(
            level="INFO",
            message=f"Initialized workflow: {self.supervisor_state.workflow_id}",
            task_id=task_id,
            context_id=context_id,
            extra={"user_request_length": len(user_request)}
        )
    
    @log_sync
    def add_agent(self, agent: BaseSubgraphAgent):
        """Add a new agent to the supervisor."""
        self.agents[agent.name] = agent
        # Rebuild graph with new agent
        self.graph = self._build_supervisor_graph()
        self.compiled_graph = self.graph.compile(checkpointer=self.memory)
        
        supervisor_logger.log_structured(
            level="INFO",
            message=f"Added agent to supervisor: {agent.name}",
            extra={"agent_name": agent.name, "total_agents": len(self.agents)}
        )
    
    @log_sync
    def remove_agent(self, agent_name: str):
        """Remove an agent from the supervisor."""
        if agent_name in self.agents:
            del self.agents[agent_name]
            # Rebuild graph without the agent
            self.graph = self._build_supervisor_graph()
            self.compiled_graph = self.graph.compile(checkpointer=self.memory)
            
            supervisor_logger.log_structured(
                level="INFO",
                message=f"Removed agent from supervisor: {agent_name}",
                extra={"agent_name": agent_name, "total_agents": len(self.agents)}
            )
        else:
            supervisor_logger.log_structured(
                level="WARNING",
                message=f"Attempted to remove non-existent agent: {agent_name}",
                extra={"agent_name": agent_name}
            )
    
    @log_sync
    def get_agent_status(self, agent_name: str) -> Optional[str]:
        """Get the status of a specific agent."""
        if agent_name in self.agents:
            return "available"
        return None
    
    @log_sync
    def list_agents(self) -> List[str]:
        """List all available agents."""
        agent_list = list(self.agents.keys())
        supervisor_logger.log_structured(
            level="DEBUG",
            message="Listed available agents",
            extra={"agent_count": len(agent_list), "agents": agent_list}
        )
        return agent_list
    
    @log_sync
    def get_agent_info(self) -> Dict[str, str]:
        """Get information about registered agents."""
        return {name: name for name in self.agents.keys()}
    
    @log_sync
    def is_ready(self) -> bool:
        """Check if the supervisor is ready for use."""
        model_ready = self.model is not None
        supervisor_ready = self.compiled_graph is not None
        agents_ready = len(self.agents) > 0
        
        # Log detailed status for debugging
        if not model_ready:
            supervisor_logger.log_structured(
                level="DEBUG",
                message="Supervisor not ready: LLM model not initialized",
                extra={"model_ready": model_ready, "supervisor_ready": supervisor_ready, "agents_ready": agents_ready}
            )
        elif not supervisor_ready:
            supervisor_logger.log_structured(
                level="DEBUG",
                message="Supervisor not ready: Supervisor graph not compiled",
                extra={"model_ready": model_ready, "supervisor_ready": supervisor_ready, "agents_ready": agents_ready}
            )
        elif not agents_ready:
            supervisor_logger.log_structured(
                level="DEBUG",
                message="Supervisor not ready: No agent subgraphs registered",
                extra={"model_ready": model_ready, "supervisor_ready": supervisor_ready, "agents_ready": agents_ready}
            )
        
        # Return true only if all components are ready
        return model_ready and supervisor_ready and agents_ready
    
    @log_sync
    def reset_state(self) -> None:
        """Reset the supervisor state."""
        self.supervisor_state = None
        supervisor_logger.log_structured(
            level="INFO",
            message="Supervisor state reset"
        )
    
    @log_sync
    def get_config_info(self) -> Dict[str, Any]:
        """Get configuration information for debugging."""
        return {
            "llm_provider": self.config_instance.get_llm_config().get('provider'),
            "llm_model": self.config_instance.get_llm_config().get('model'),
            "supervisor_config": self.supervisor_config,
            "agent_count": len(self.agents),
            "ready": self.is_ready()
        }

    def _validate_all_agents_completion(self, item: dict) -> bool:
        """
        Validate that ALL agents in the supervisor have completed their individual flows.
        Only return True when the entire supervisor workflow is complete.
        
        Args:
            item: The state update from any agent in the supervisor
            
        Returns:
            bool: True if ALL agents have completed their flows, False otherwise
        """
        try:
            # Check if this is a handoff tool execution (final completion signal)
            if 'planner_data' in item and item.get('planner_data'):
                supervisor_logger.log_structured(
                    level="INFO",
                    message="Handoff tool executed - supervisor workflow complete",
                    extra={
                        "has_planner_data": True,
                        "planner_data_keys": list(item.get('planner_data', {}).keys()),
                        "validation_result": "PASS"
                    }
                )
                return True
            
            # Get list of attached agents from supervisor configuration
            attached_agents = self._get_attached_agents()
            
            # Validate each agent type individually
            agent_completion_status = {}
            agent_status_data = {}
            for agent_type in attached_agents:
                if agent_type == "planner":
                    is_complete, planning_status = self._validate_planner_agent_completion(item)
                    agent_completion_status["planner"] = is_complete
                    agent_status_data["planner"] = planning_status
                elif agent_type == "generator":
                    agent_completion_status["generator"] = self._validate_generator_agent_completion(item)
                    agent_status_data["generator"] = {}
                elif agent_type == "validator":
                    agent_completion_status["validator"] = self._validate_validator_agent_completion(item)
                    agent_status_data["validator"] = {}
                elif agent_type == "editor":
                    agent_completion_status["editor"] = self._validate_editor_agent_completion(item)
                    agent_status_data["editor"] = {}
                # Add more agent types as needed
            
            # Check if ALL agents are completed
            all_agents_completed = all(agent_completion_status.values())
            
            supervisor_logger.log_structured(
                level="INFO",
                message="Multi-agent completion validation",
                extra={
                    "attached_agents": attached_agents,
                    "agent_completion_status": agent_completion_status,
                    "agent_status_data_keys": {agent: list(data.keys()) for agent, data in agent_status_data.items()},
                    "all_agents_completed": all_agents_completed,
                    "validation_result": "PASS" if all_agents_completed else "CONTINUE"
                }
            )
            
            return all_agents_completed
            
        except Exception as e:
            supervisor_logger.log_structured(
                level="ERROR",
                message=f"Error validating all agents completion: {e}",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "item_keys": list(item.keys()) if isinstance(item, dict) else "not_dict"
                }
            )
            return False

    def _get_attached_agents(self) -> List[str]:
        """
        Get list of agent types attached to this supervisor.
        
        Returns:
            List of agent type names
        """
        # For now, we know we have planner agent
        # In the future, this could be dynamically determined from self.agents
        return ["planner"]  # Add more as needed: ["planner", "generator", "validator", "editor"]

    def _validate_planner_agent_completion(self, item: dict) -> tuple[bool, dict]:
        """
        Validate that the planner agent has completed all its required phases.
        
        Args:
            item: The state update from the supervisor
            
        Returns:
            tuple: (is_complete, planning_status) where is_complete is bool and planning_status contains all planning information
        """
        try:
            planning_state = item.get('planning_workflow_state')
            if not planning_state:
                supervisor_logger.log_structured(
                    level="WARNING",
                    message="No planning_workflow_state found for planner validation",
                    extra={"item_keys": list(item.keys()) if isinstance(item, dict) else "not_dict"}
                )
                return False, {}
            
            # Extract completion flags
            if hasattr(planning_state, 'requirements_complete'):
                requirements_complete = planning_state.requirements_complete
                execution_complete = planning_state.execution_complete
                planning_complete = planning_state.planning_complete
                error_occurred = planning_state.error_occurred
            elif isinstance(planning_state, dict):
                requirements_complete = planning_state.get('requirements_complete', False)
                execution_complete = planning_state.get('execution_complete', False)
                planning_complete = planning_state.get('planning_complete', False)
                error_occurred = planning_state.get('error_occurred', False)
            else:
                supervisor_logger.log_structured(
                    level="WARNING",
                    message="Unexpected planning_workflow_state type for planner validation",
                    extra={"planning_state_type": type(planning_state).__name__}
                )
                return False, {}
            
            # Check all required planner phases are complete
            planner_completion_metrics = {
                'requirements_complete': requirements_complete,
                'execution_complete': execution_complete,
                'planning_complete': planning_complete,
                'no_errors': not error_occurred
            }
            
            # All planner phases must be complete and no errors
            planner_phases_complete = all(planner_completion_metrics.values())
            
            # If phases are complete, also validate that we have the corresponding data
            if planner_phases_complete:
                data_validation, data_values = self._validate_planner_data_availability(item)
                if not data_validation:
                    supervisor_logger.log_structured(
                        level="WARNING",
                        message="All planner phases completed but missing corresponding data",
                        extra={
                            "planner_completion_metrics": planner_completion_metrics,
                            "data_validation_failed": True,
                            "available_data_keys": list(data_values.keys())
                        }
                    )
                    return False, {}
                
                # Store the actual data values for later use
                supervisor_logger.log_structured(
                    level="INFO",
                    message="Planner data validation successful - data values available",
                    extra={
                        "planner_completion_metrics": planner_completion_metrics,
                        "data_validation_passed": True,
                        "available_data_keys": list(data_values.keys()),
                        "data_values_preview": {
                            key: str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                            for key, value in data_values.items()
                        }
                    }
                )
            
            # Prepare comprehensive planning status
            planning_status = {
                "completion_metrics": planner_completion_metrics,
                "phases_complete": planner_phases_complete,
                "planning_state": planning_state,
                "data_values": data_values if planner_phases_complete else {},
                "validation_result": "PASS" if planner_phases_complete else "CONTINUE"
            }
            
            supervisor_logger.log_structured(
                level="INFO",
                message="Planner agent completion validation",
                extra={
                    "planner_completion_metrics": planner_completion_metrics,
                    "planner_phases_complete": planner_phases_complete,
                    "planner_fully_complete": planner_phases_complete,
                    "validation_result": "PASS" if planner_phases_complete else "CONTINUE",
                    "planning_status_keys": list(planning_status.keys())
                }
            )
            
            return planner_phases_complete, planning_status
            
        except Exception as e:
            supervisor_logger.log_structured(
                level="ERROR",
                message=f"Error validating planner agent completion: {e}",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "item_keys": list(item.keys()) if isinstance(item, dict) else "not_dict"
                }
            )
            return False, {}

    def _validate_planner_data_availability(self, item: dict) -> tuple[bool, dict]:
        """
        Validate that all required planner data is available when phases are completed.
        
        Args:
            item: The state update from the planner sub-supervisor
            
        Returns:
            tuple: (is_valid, data_dict) where is_valid is bool and data_dict contains the actual data values
        """
        try:
            # Extract planning workflow state
            planning_state = item.get('planning_workflow_state')
            if not planning_state:
                supervisor_logger.log_structured(
                    level="WARNING",
                    message="No planning_workflow_state found for data validation",
                    extra={"item_keys": list(item.keys()) if isinstance(item, dict) else "not_dict"}
                )
                return False, {}
            
            # Extract actual data values - check both top level and planning_state
            data_values = {}
            missing_data = []
            
            # Get requirements_data - check top level first, then planning_state
            requirements_data = None
            if 'requirements_data' in item and item['requirements_data'] is not None and item['requirements_data'] != "":
                requirements_data = item['requirements_data']
            elif hasattr(planning_state, 'requirements_data'):
                requirements_data = getattr(planning_state, 'requirements_data')
            elif isinstance(planning_state, dict) and 'requirements_data' in planning_state:
                requirements_data = planning_state['requirements_data']
            
            if requirements_data is not None and requirements_data != "":
                data_values['requirements_data'] = requirements_data
            else:
                missing_data.append('requirements_data')
            
            # Get execution_data - check top level first, then planning_state
            execution_data = None
            if 'execution_data' in item and item['execution_data'] is not None and item['execution_data'] != "":
                execution_data = item['execution_data']
            elif hasattr(planning_state, 'execution_data'):
                execution_data = getattr(planning_state, 'execution_data')
            elif isinstance(planning_state, dict) and 'execution_data' in planning_state:
                execution_data = planning_state['execution_data']
            
            if execution_data is not None and execution_data != "":
                data_values['execution_data'] = execution_data
            else:
                missing_data.append('execution_data')
            
            # Log data availability with actual values
            supervisor_logger.log_structured(
                level="INFO",
                message="Planner data availability validation",
                extra={
                    "available_data_keys": list(data_values.keys()),
                    "missing_data": missing_data,
                    "all_data_available": len(missing_data) == 0,
                    "data_values_preview": {
                        key: str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                        for key, value in data_values.items()
                    }
                }
            )
            
            # Return validation result and actual data values
            is_valid = len(missing_data) == 0
            return is_valid, data_values
            
        except Exception as e:
            supervisor_logger.log_structured(
                level="ERROR",
                message="Error validating planner data availability",
                extra={"error": str(e), "error_type": type(e).__name__}
            )
            return False, {}

    def _validate_generator_agent_completion(self, item: dict) -> bool:
        """
        Validate that the generator agent has completed its work.
        
        Args:
            item: The state update from the supervisor
            
        Returns:
            bool: True if generator agent is complete, False otherwise
        """
        # TODO: Implement generator agent completion validation
        # For now, return True as we don't have generator agent yet
        return True

    def _validate_validator_agent_completion(self, item: dict) -> bool:
        """
        Validate that the validator agent has completed its work.
        
        Args:
            item: The state update from the supervisor
            
        Returns:
            bool: True if validator agent is complete, False otherwise
        """
        # TODO: Implement validator agent completion validation
        # For now, return True as we don't have validator agent yet
        return True

    def _validate_editor_agent_completion(self, item: dict) -> bool:
        """
        Validate that the editor agent has completed its work.
        
        Args:
            item: The state update from the supervisor
            
        Returns:
            bool: True if editor agent is complete, False otherwise
        """
        # TODO: Implement editor agent completion validation
        # For now, return True as we don't have editor agent yet
        return True


# Factory function for easy supervisor creation
def create_supervisor_agent(
    agents: List[BaseSubgraphAgent],
    config: Optional[Config] = None,
    custom_config: Optional[Dict[str, Any]] = None,
    prompt_template: Optional[str] = None,
    name: str = "supervisor-agent"
) -> CustomSupervisorAgent:
    """
    Create a supervisor agent with the given agents using centralized configuration.
    
    Args:
        agents: List of subgraph agents to orchestrate
        config: Configuration instance (defaults to new Config())
        custom_config: Optional custom configuration to override defaults
        prompt_template: Custom prompt template
        name: Agent name for identification
        
    Returns:
        Configured supervisor agent
    """
    return CustomSupervisorAgent(
        agents=agents,
        config=config,
        custom_config=custom_config,
        prompt_template=prompt_template,
        name=name
    )
