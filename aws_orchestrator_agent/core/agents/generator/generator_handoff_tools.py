"""
Generator Handoff Tools for returning to main supervisor.

This module implements handoff tools that allow the generator swarm
to return control back to the main supervisor with generated artifacts.
"""

from typing import Annotated, Dict, Any
from langchain_core.tools import tool, BaseTool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.prebuilt import InjectedState

from .generator_state import GeneratorSwarmState
from aws_orchestrator_agent.utils.logger import AgentLogger

# Create logger
generator_handoff_logger = AgentLogger("GENERATOR_HANDOFF")


def create_handoff_to_generator_complete() -> BaseTool:
    """
    Create a handoff tool for the generator swarm to return to the main supervisor.
    
    Returns:
        BaseTool: Handoff tool for returning to supervisor
    """
    
    @tool("handoff_to_generator_complete", description="Signal completion of generator swarm and return to main supervisor")
    def handoff_to_generator_complete(
        completion_summary: Annotated[str, "Summary of what was generated and completed"],
        generated_artifacts: Annotated[Dict[str, Any], "Summary of all generated artifacts"],
        state: Annotated[GeneratorSwarmState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId]
    ) -> Command:
        """
        Handoff to main supervisor with generator completion.
        
        This tool:
        1. Extracts all generated artifacts from the generator swarm state
        2. Creates a completion summary
        3. Returns a Command to transfer control back to the main supervisor
        4. Passes all generated artifacts and completion status
        """
        
        try:
            generator_handoff_logger.log_structured(
                level="INFO",
                message="Generator swarm completion handoff initiated",
                extra={
                    "completion_summary": completion_summary[:100] + "..." if len(completion_summary) > 100 else completion_summary,
                    "generated_artifacts_keys": list(generated_artifacts.keys()) if generated_artifacts else [],
                    "stage_status": state.get("stage_status", "unknown"),
                    "current_stage": state.get("current_stage", "unknown")
                }
            )
            
            # Extract all generated artifacts from agent workspaces
            agent_workspaces = state.get("agent_workspaces", {})
            
            # Collect all generated artifacts
            generated_resources = agent_workspaces.get("resource_configuration_agent", {}).get("generated_resources", [])
            generated_variables = agent_workspaces.get("variable_definition_agent", {}).get("generated_variables", [])
            generated_data_sources = agent_workspaces.get("data_source_agent", {}).get("generated_data_sources", [])
            generated_locals = agent_workspaces.get("local_values_agent", {}).get("generated_locals", [])
            generated_outputs = agent_workspaces.get("output_definition_agent", {}).get("generated_outputs", [])
            
            # Extract planning context for metadata
            planning_context = state.get("planning_context", {})
            
            # Create comprehensive generation data
            generation_data = {
                "generated_resources": generated_resources,
                "generated_variables": generated_variables,
                "generated_data_sources": generated_data_sources,
                "generated_locals": generated_locals,
                "generated_outputs": generated_outputs,
                "module_name": planning_context.get("module_name", "terraform-module"),
                "service_name": planning_context.get("service_name", "Unknown Service"),
                "target_environment": planning_context.get("target_environment", "prod"),
                "stage_progress": state.get("stage_progress", {}),
                "agent_status_matrix": state.get("agent_status_matrix", {}),
                "generation_complete": True,
                "total_artifacts": len(generated_resources) + len(generated_variables) + 
                                 len(generated_data_sources) + len(generated_locals) + len(generated_outputs),
                "completion_summary": completion_summary,
                "generated_artifacts_summary": generated_artifacts
            }
            
            # Create tool message
            tool_message = ToolMessage(
                content=f"Generator swarm completed: {completion_summary}",
                name="handoff_to_generator_complete",
                tool_call_id=tool_call_id
            )
            
            # Create state updates for supervisor
            supervisor_updates = {
                "messages": [*state.get("messages", []), tool_message],
                "generation_data": generation_data,
                "generated_module_ref": f"terraform-module-{planning_context.get('module_name', 'unknown')}",
                "question": None,  # Generator swarm doesn't typically ask questions
                "current_agent": None,  # Generation complete, supervisor decides next step
                "status": "completed",
                "workflow_state": {
                    "generation_complete": True,
                    "next_agent": "validation_agent",  # Default next step
                }
            }
            
            generator_handoff_logger.log_structured(
                level="INFO",
                message="Generator swarm completion handoff prepared",
                extra={
                    "generation_data_keys": list(generation_data.keys()),
                    "total_artifacts": generation_data["total_artifacts"],
                    "module_name": generation_data["module_name"],
                    "supervisor_updates_keys": list(supervisor_updates.keys())
                }
            )
            
            # Return Command to transfer control back to main supervisor
            return Command(
                goto="supervisor",  # Return to main supervisor
                graph=Command.PARENT,  # Go to parent graph
                update=supervisor_updates
            )
            
        except Exception as e:
            generator_handoff_logger.log_structured(
                level="ERROR",
                message="Generator swarm completion handoff failed",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "completion_summary": completion_summary[:100] if completion_summary else "None"
                }
            )
            # Return safe fallback command
            return Command(
                goto="supervisor",
                graph=Command.PARENT,
                update={
                    "messages": [*state.get("messages", []), ToolMessage(
                        content="Generator swarm completed with errors",
                        name="handoff_to_generator_complete",
                        tool_call_id=tool_call_id
                    )],
                    "generation_data": {
                        "generated_resources": [],
                        "generated_variables": [],
                        "generated_data_sources": [],
                        "generated_locals": [],
                        "generated_outputs": [],
                        "generation_complete": False,
                        "total_artifacts": 0,
                        "error": str(e)
                    },
                    "status": "error",
                    "current_agent": None
                }
            )
    
    return handoff_to_generator_complete
