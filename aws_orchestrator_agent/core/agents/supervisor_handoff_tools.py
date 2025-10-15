"""
Custom Handoff Tools for Supervisor using langgraph-supervisor.

This module implements custom handoff tools that:
- Pass session_id and task_id to agents
- Maintain our existing state structure
- Work with langgraph-supervisor's create_supervisor() function
"""

from typing import Annotated, Dict, Any, Optional
from langchain_core.tools import tool, BaseTool, InjectedToolCallId
from langchain_core.messages import ToolMessage, HumanMessage
from langgraph.types import Command, Send
from langgraph.prebuilt import InjectedState
from langgraph_supervisor.handoff import METADATA_KEY_HANDOFF_DESTINATION
from aws_orchestrator_agent.core.agents.types import StateTransformer
from aws_orchestrator_agent.utils.logger import AgentLogger

# Create logger
handoff_logger = AgentLogger("SUPERVISOR_HANDOFF")


def create_custom_handoff_tool(*, agent_name: str, name: str | None, description: str | None) -> BaseTool:
    """
    Create a custom handoff tool that passes session_id and task_id to agents.
    
    Args:
        agent_name: Name of the target agent
        name: Tool name (if None, will be auto-generated)
        description: Tool description (if None, will be auto-generated)
        
    Returns:
        BaseTool: Custom handoff tool
    """
    
    # Auto-generate name and description if not provided
    if name is None:
        name = f"transfer_to_{agent_name}"
    if description is None:
        description = f"Transfer task to {agent_name}"

    @tool(name, description=description)
    def handoff_to_agent(
        # Task description for the LLM to populate
        task_description: Annotated[str, "Detailed description of what the next agent should do, including all of the relevant context."],
        # Injected state from the supervisor
        state: Annotated[Any, InjectedState],
        # Injected tool call ID
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        """
        Handoff to a specific agent with session_id and task_id.
        
        This tool:
        1. Extracts session_id and task_id from the supervisor state
        2. Creates a tool message for the handoff
        3. Returns a Command to transfer control to the target agent
        4. Passes all necessary context including session_id and task_id
        """
        
        # Log the handoff attempt
        handoff_logger.log_structured(
            level="DEBUG",
            message=f"Handoff tool called for {agent_name}",
            extra={
                "agent_name": agent_name,
                "task_description": task_description,
                "state_keys": list(state.keys()) if isinstance(state, dict) else "not_dict",
                "state_type": type(state).__name__,
            }
        )
        
        # Extract session_id and task_id from state
        # Handle both Pydantic models and dictionaries
        session_id = None
        task_id = None
        user_request = ""
        
        # Convert Pydantic model to dict if needed
        if hasattr(state, 'model_dump'):
            state_dict = state.model_dump()
        elif hasattr(state, 'dict'):
            state_dict = state.dict()
        elif isinstance(state, dict):
            state_dict = state
        else:
            # Fallback: try to access attributes directly
            state_dict = {}
            if hasattr(state, 'session_id'):
                state_dict['session_id'] = state.session_id
            if hasattr(state, 'task_id'):
                state_dict['task_id'] = state.task_id
            if hasattr(state, 'user_request'):
                state_dict['user_request'] = state.user_request
            if hasattr(state, 'messages'):
                state_dict['messages'] = state.messages
        
        # Try to extract from state directly first
        session_id = state_dict.get("session_id")
        task_id = state_dict.get("task_id")
        user_request = state_dict.get("user_request", "")
        
        # If not found in state, try to extract from messages
        if session_id is None or task_id is None or not user_request:
            messages = state_dict.get("messages", [])
            for message in messages:
                if hasattr(message, 'additional_kwargs'):
                    metadata = message.additional_kwargs
                    if session_id is None and 'session_id' in metadata:
                        session_id = metadata['session_id']
                    if task_id is None and 'task_id' in metadata:
                        task_id = metadata['task_id']
                    if not user_request and 'user_request' in metadata:
                        user_request = metadata['user_request']
                elif hasattr(message, 'metadata'):
                    metadata = message.metadata
                    if session_id is None and 'session_id' in metadata:
                        session_id = metadata['session_id']
                    if task_id is None and 'task_id' in metadata:
                        task_id = metadata['task_id']
                    if not user_request and 'user_request' in metadata:
                        user_request = metadata['user_request']
        
        # Log extracted values
        handoff_logger.log_structured(
            level="DEBUG",
            message=f"Extracted session_id, task_id, and user_request from state",
            extra={
                "extracted_session_id": session_id,
                "extracted_task_id": task_id,
                "extracted_user_request": user_request,
                "state_has_session_id": "session_id" in state_dict,
                "state_has_task_id": "task_id" in state_dict,
                "state_has_user_request": "user_request" in state_dict,
            }
        )
        
        # Create tool message for the handoff
        # Add error handling for missing tool_call_id
        if not tool_call_id:
            handoff_logger.log_structured(
                level="ERROR",
                message=f"Missing tool_call_id for {agent_name} handoff",
                extra={
                    "agent_name": agent_name,
                    "tool_call_id": tool_call_id,
                    "tool_call_id_type": type(tool_call_id).__name__,
                }
            )
            # Use a fallback tool_call_id
            tool_call_id = f"fallback_{agent_name}_{session_id[:8] if session_id else 'unknown'}"
        
        # Create tool message for the handoff
        # Ensure tool_call_id is properly set
        handoff_logger.log_structured(
            level="DEBUG",
            message=f"Creating tool message for {agent_name}",
            extra={
                "agent_name": agent_name,
                "tool_call_id": tool_call_id,
                "tool_call_id_type": type(tool_call_id).__name__,
                "tool_call_id_length": len(tool_call_id) if tool_call_id else 0,
            }
        )
        
        try:
            tool_message = ToolMessage(
                content=f"Successfully transferred to {agent_name}",
                name=name,
                tool_call_id=tool_call_id,
            )
            
            # Verify the tool message was created correctly
            handoff_logger.log_structured(
                level="DEBUG",
                message=f"ToolMessage created successfully for {agent_name}",
                extra={
                    "agent_name": agent_name,
                    "message_type": type(tool_message).__name__,
                    "has_tool_call_id": hasattr(tool_message, 'tool_call_id'),
                    "tool_call_id_value": getattr(tool_message, 'tool_call_id', 'NOT_FOUND'),
                }
            )
            
        except Exception as e:
            handoff_logger.log_structured(
                level="ERROR",
                message=f"Failed to create ToolMessage for {agent_name}",
                extra={
                    "agent_name": agent_name,
                    "tool_call_id": tool_call_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
            )
            # Fallback: create a simple message without tool_call_id
            tool_message = ToolMessage(
                content=f"Successfully transferred to {agent_name}",
                name=name,
                tool_call_id=tool_call_id,
            )
        
        # Get messages from state
        messages = state_dict.get("messages", [])
        
        # Extract planner data from supervisor state
        planner_data = state_dict.get("planner_data")
        # Extract execution plan from the nested structure
        # execution_plan = {}
        # if planner_data and "execution_data" in planner_data:
        #     execution_data = planner_data["execution_data"]
        #     if execution_data and "execution_plan_data" in execution_data:
        #         execution_plan_data = execution_data["execution_plan_data"]
        #         if execution_plan_data and "execution_plans" in execution_plan_data:
        #             execution_plans = execution_plan_data["execution_plans"]
        #             if execution_plans and len(execution_plans) > 0:
        #                 execution_plan = execution_plans[0]  # Use the first execution plan
        
        # Use proper state transformation for planner handoff
        if agent_name == "planner_sub_supervisor":
            # Create a proper SupervisorState object for transformation
            # Don't include the tool_message in the transformation to avoid tool_call_id issues
            supervisor_state = type('SupervisorState', (), {
                'user_request': user_request if user_request else task_description,
                'session_id': session_id,
                'task_id': task_id,
                'messages': messages,  # Don't include tool_message here
                'workspace_ref': state_dict.get("workspace_ref"),
                'terraform_context': state_dict.get("terraform_context", {}),
            })()
            
            # Transform to proper PlannerSupervisorState
            planner_state = StateTransformer.supervisor_to_planner(supervisor_state)
            
            # Convert to dict for Command update
            state_update = planner_state.model_dump()
            
            # Add the tool message separately to avoid tool_call_id issues
            state_update["messages"] = messages + [tool_message]
            
            # No need to remove workflow_state since we're using planning_workflow_state (distinct field name)
            handoff_logger.log_structured(
                level="DEBUG",
                message=f"Using distinct field name planning_workflow_state to avoid supervisor merge",
                extra={
                    "agent_name": agent_name,
                    "has_planning_workflow_state": "planning_workflow_state" in state_update,
                    "has_workflow_state": "workflow_state" in state_update,
                    "final_state_keys": list(state_update.keys()),
                }
            )
            
            handoff_logger.log_structured(
                level="DEBUG",
                message=f"Using state transformation for {agent_name}",
                extra={
                    "agent_name": agent_name,
                    "transformation_method": "StateTransformer.supervisor_to_planner",
                    "state_update_keys": list(state_update.keys()),
                    "has_workflow_state": "workflow_state" in state_update,
                    "workflow_state_type": type(state_update.get("workflow_state")).__name__ if state_update.get("workflow_state") else "None",
                    "messages_count": len(state_update.get("messages", [])),
                }
            )
        elif agent_name == "generator_swarm":
            # For generator_swarm, pass only what StateTransformer.supervisor_to_generator_swarm needs
            # The transformation node wrapper will handle the state transformation
            state_update = {
                "messages": messages + [tool_message],
                "session_id": session_id,
                "task_id": task_id,
                "planner_data": state_dict.get("planner_data"),
            }
            # Log the transformation node wrapper approach
            handoff_logger.log_structured(
                level="DEBUG",
                message=f"Using transformation node wrapper approach for {agent_name}",
                extra={
                    "agent_name": agent_name,
                    "transformation_method": "transformation_node_wrapper",
                    "state_update_keys": list(state_update.keys()),
                    "messages_count": len(state_update.get("messages", [])),
                    "has_planner_data": state_update.get("planner_data") is not None,
                    "note": "Minimal state passed - StateTransformer only needs planner_data, session_id, task_id",
                    "architecture": "transformation_node_wrapper"
                }
            )
        elif agent_name == "writer_react_agent":
            # For writer_react_agent, pass only what StateTransformer.supervisor_to_writer_react_agent needs
            writer_react_state = StateTransformer.supervisor_to_writer_react(state_dict)
            state_update = writer_react_state.model_dump()
            state_update["messages"] = messages + [tool_message]
            # Log the transformation node wrapper approach
            handoff_logger.log_structured(
                level="DEBUG",
                message=f"Using minimal state for {agent_name}",
                extra={
                    "agent_name": agent_name,
                    "transformation_method": "minimal_state",
                    "state_update_keys": list(state_update.keys()),
                    "messages_count": len(state_update.get("messages", [])),
                    "has_generation_data": state_update.get("generation_data") is not None,
                    "note": "Minimal state passed - StateTransformer only needs generation_data, session_id, task_id",
                    "architecture": "minimal_state"
                }
            )
        else:
            # For other agents, use the existing approach
            # First, create a clean state dict without workflow_state
            clean_state_dict = {k: v for k, v in state_dict.items() if k != "workflow_state"}
            
            # Create the state update with only the fields we want to pass
            state_update = {
                "messages": messages + [tool_message],
                "active_agent": agent_name,
                "task_description": task_description,
                # Pass our custom fields
                "session_id": session_id,
                "task_id": task_id,
                # Pass the extracted user_request
                "user_request": user_request if user_request else task_description,
                "status": clean_state_dict.get("status", "in_progress"),
                # Explicitly exclude workflow_state to prevent schema conflicts
                # Each agent should manage its own workflow state independently
            }
        
        # Log what we're excluding (only for non-transformed agents)
        if agent_name not in ["planner_sub_supervisor", "generator_swarm", "writer_react_agent"] and "workflow_state" in state_dict:
            handoff_logger.log_structured(
                level="DEBUG",
                message=f"Excluding workflow_state from handoff to {agent_name}",
                extra={
                    "agent_name": agent_name,
                    "workflow_state_type": type(state_dict["workflow_state"]).__name__,
                    "excluded_keys": ["workflow_state"],
                    "included_keys": list(state_update.keys())
                }
            )
        
        # Log the final state update
        handoff_logger.log_structured(
            level="DEBUG",
            message=f"Handing off to {agent_name} - using nested state approach for generator_swarm",
            extra={
                "agent_name": agent_name,
                "state_update_keys": list(state_update.keys()),
                "final_session_id": state_update.get("session_id"),
                "final_task_id": state_update.get("task_id"),
                "note": "nested generator_state approach - generator_state created by input_transform()" if agent_name == "generator_swarm" else "workflow_state excluded to prevent schema conflicts",
                "has_workflow_state_in_original": "workflow_state" in state_dict,
                "architecture": "nested_state_approach" if agent_name == "generator_swarm" else "standard_handoff"
            }
        )
        
        # Return Command to transfer control to the target agent
        # Use explicit state update to ensure only the fields we want are passed
        
        # Use Command for all agents - simple routing to wrapper function
        return Command(
            goto=agent_name,
            graph=Command.PARENT,
            update=state_update,
        )

    # Set metadata for langgraph-supervisor
    handoff_to_agent.metadata = {METADATA_KEY_HANDOFF_DESTINATION: agent_name}
    
    return handoff_to_agent


def create_handoff_tools_for_agents(agent_names: list[str]) -> list[BaseTool]:
    """
    Create handoff tools for multiple agents.
    
    Args:
        agent_names: List of agent names to create handoff tools for
        
    Returns:
        List[BaseTool]: List of handoff tools
    """
    tools = []
    
    for agent_name in agent_names:
        tool = create_custom_handoff_tool(
            agent_name=agent_name,
            name=None,  # Auto-generate name
            description=None  # Auto-generate description
        )
        tools.append(tool)
        
        handoff_logger.log_structured(
            level="INFO",
            message=f"Created handoff tool for {agent_name}",
            extra={
                "agent_name": agent_name,
                "tool_name": tool.name,
                "tool_description": tool.description,
            }
        )
    
    return tools
