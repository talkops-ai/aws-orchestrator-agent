import json
import ast
from typing import Annotated, Dict, Any, List, Optional
from langchain_core.tools import tool
from langgraph.types import Command
from langchain_core.messages import ToolMessage, HumanMessage
from langgraph.prebuilt import InjectedState
from langchain_core.tools import InjectedToolCallId, tool
from langgraph_supervisor.handoff import METADATA_KEY_HANDOFF_DESTINATION
from .generator_state import GeneratorSwarmState, DependencyType, GeneratorAgentStatus
from .generator_state_controller import GeneratorStageController
from .global_state import get_current_state, update_current_state
from aws_orchestrator_agent.utils.logger import AgentLogger
import datetime


class GeneratorStageHandoffManager:
    """Manages sophisticated handoffs with dependency resolution and context filtering"""
    
    def __init__(self):
        self.logger = AgentLogger("GeneratorStageHandoffManager")
        # self.planner_state = get_current_state()
    
    def create_dependency_aware_handoff_tool(
        self, 
        target_agent: str,
        active_agent: str,
        dependency_type: DependencyType,
        description: str
    ):
        """Create intelligent handoff tools that understand dependencies"""
        
        @tool(f"handoff_to_{target_agent}_{dependency_type.value}", description=description)
        def dependency_handoff_tool(
            task_description: Annotated[str, "Specific task for target agent"],
            dependency_data: Annotated[Dict[str, Any], "Structured dependency data"],
            state: Annotated[Any, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],
            priority_level: Annotated[int, "Priority: 1=low, 5=critical"] = 3,
            blocking: Annotated[bool, "Whether source agent should wait for completion"] = True
        ) -> Command:
            """Handoff tool for coordinating dependencies between generator swarm agents.

Enables the current agent to delegate specific tasks to target agents when 
dependencies are discovered during resource generation. Updates state, tracks 
dependencies, and coordinates agent transitions.

Args:
    task_description (str): Specific, actionable task for the target agent.
        Example: "Define VPC variables including CIDR block and DNS settings"
    
    dependency_data (dict): Structured dependency information for the target agent.
        For variable dependencies:
        - variables: List with name, type, description, default values
        - validation_rules: Validation requirements (optional)
        - usage_context: How variables will be used (optional)
        Example: {
            "variables": [
                {"name": "cidr_block", "type": "string", "description": "VPC CIDR", "default": "10.0.0.0/16"},
                {"name": "enable_dns", "type": "bool", "description": "Enable DNS", "default": true}
            ]
        }
    
    state: LangGraph state with swarm context. Automatically injected.
    
    tool_call_id (str): Unique identifier. Automatically provided by LangGraph.
    
    priority_level (int): Priority from 1 (low) to 5 (critical). Determines handoff order.
        - 5: Critical variable dependencies (blocking)
        - 4: High local value dependencies (blocking)  
        - 3: Medium data source dependencies (non-blocking)
        - 2-1: Optional/minimal dependencies
    
    blocking (bool): Whether source agent waits for target completion.
        True for critical dependencies, False for parallel processing.

Returns:
    Command: LangGraph command to transition to target agent.

Raises:
    ValueError: Missing or invalid required parameters.
    RuntimeError: State update or agent transition failure.

Example:
    result = handoff_to_variable_definition_agent_resource_to_variable(
        task_description="Define VPC configuration variables",
        dependency_data={
            "variables": [
                {"name": "vpc_cidr", "type": "string", "description": "VPC CIDR block"},
                {"name": "enable_dns", "type": "bool", "description": "Enable DNS support"}
            ]
        },
        priority_level=5,
        blocking=True
    )
        """
            try:
                previous_state = get_current_state()
                tool_response = None
                
                # Extract discovered dependencies and handoff recommendations from global state
                discovered_dependencies = []
                handoff_recommendations = []
                
                if previous_state and "agent_workspaces" in previous_state:
                    resource_workspace = previous_state["agent_workspaces"].get(active_agent, {})
                    discovered_dependencies = resource_workspace.get("pending_dependencies", [])
                    handoff_recommendations = resource_workspace.get("handoff_recommendations", [])

                source_agent = active_agent
                
                self.logger.log_structured(
                    level="INFO",
                    message="Creating dependency handoff",
                    extra={
                        "source_agent": source_agent,
                        "target_agent": target_agent,
                        "dependency_type": dependency_type.value,
                        "priority_level": priority_level,
                        "blocking": blocking,
                        "task_description": task_description[:100] + "..." if len(task_description) > 100 else task_description
                    }
                )
                
                # Transform handoff recommendations data for target agent context
                transformed_dependency_data = discovered_dependencies
                dependencies = []  # Initialize dependencies as empty list
                context_payload = {}  # Initialize context_payload as empty dict
                
                if handoff_recommendations:
                    # Find the specific recommendation for our target agent
                    target_rec = None
                    for rec in handoff_recommendations:
                        if rec.get("target_agent") == target_agent:
                            target_rec = rec
                            break
                    
                    if target_rec:
                        context_payload = target_rec.get("context_payload", {})
                        dependencies = context_payload.get("dependencies", [])
                        # affected_resources = context_payload.get("affected_resources", [])
                
                # Create dependency request (keep original dependency_data for tracking)
                dependency_request = {
                    "id": f"{source_agent}_{target_agent}_{int(datetime.datetime.now().timestamp())}",
                    "source_agent": source_agent,
                    "target_agent": target_agent,
                    "dependency_type": dependency_type.value,
                    "task_description": task_description,
                    "dependency_data": dependencies,  # Keep original for tracking
                    "priority_level": priority_level,
                    "blocking": blocking,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "status": "pending"
                }
            
                # Update dependency tracking (defensive handling for missing keys)
                current_pending_deps = previous_state.get("pending_dependencies", {})
                updated_pending_deps = {
                    **current_pending_deps,
                    target_agent: [
                        *current_pending_deps.get(target_agent, []),
                        dependency_request
                    ]
                }
                
                # Update dependency graph (defensive handling for missing keys)
                current_dep_graph = previous_state.get("dependency_graph", {})
                updated_dep_graph = {
                    **current_dep_graph,
                    target_agent: {
                        *current_dep_graph.get(target_agent, set()),
                        source_agent
                    }
                }
                
                # Update agent status (defensive handling for missing keys)
                current_status_matrix = previous_state.get("agent_status_matrix", {})
                # Use controller utility to ensure complete status matrix and merge properly
                controller = GeneratorStageController()
                status_updates = {target_agent: GeneratorAgentStatus.ACTIVE}
                if blocking:
                    status_updates[source_agent] = GeneratorAgentStatus.WAITING
                
                updated_status_matrix = controller.merge_status_matrix(
                    current_status_matrix,
                    status_updates
                )
                
                # Create context for target agent with transformed data
                # target_context = self.create_target_agent_context(
                #     state, dependency_request, target_agent, transformed_dependency_data
                # )
                target_context = context_payload
                # Create tool message
                tool_message = ToolMessage(
                    content=f"Dependency handoff to {target_agent}: {task_description}",
                    name=f"handoff_to_{target_agent}_{dependency_type.value}",
                    tool_call_id=tool_call_id
                )
                
                self.logger.log_structured(
                    level="INFO",
                    message="Dependency handoff completed",
                    extra={
                        "dependency_id": dependency_request["id"],
                        "source_agent": source_agent,
                        "target_agent": target_agent,
                        "blocking": blocking,
                        "pending_deps_count": len(updated_pending_deps.get(target_agent, []))
                    }
                )
                
                # For create_react_agent compatibility, return ToolMessage with handoff metadata
                # The langgraph-swarm system should handle the handoff routing from metadata
                # Create human message for clear agent instruction
                # Extract variable names from requirement_details
                # variable_names = [dep.get('requirement_details', {}).get('name', '') for dep in target_context.get('dependencies', []) if dep.get('requirement_details', {}).get('name')]
                
                # human_message_for_agent = HumanMessage(
                #     content=f"{task_description}. Required variables: {', '.join(variable_names)}"
                # )
                
                tool_message_with_handoff = ToolMessage(
                    content=f"Dependency handoff to {target_agent}: {task_description}. Required Dependencies: {dependency_data}",
                    name=f"handoff_to_{target_agent}_{dependency_type.value}",
                    tool_call_id=tool_call_id,
                    metadata={
                        "handoff_destination": target_agent,
                        "command_type": "goto",
                        "state_updates": {
                            "active_agent": target_agent,
                            "agent_status_matrix": updated_status_matrix,
                            "pending_dependencies": updated_pending_deps,
                            "dependency_graph": updated_dep_graph,
                            "agent_workspaces": {
                                **previous_state.get("agent_workspaces", {}),
                                target_agent: {
                                    **previous_state.get("agent_workspaces", {}).get(target_agent, {}),
                                    "current_task": dependency_request,
                                    "context": target_context
                                }
                            },
                            "handoff_queue": [
                                *previous_state.get("handoff_queue", []),
                                dependency_request
                            ]
                        }
                    }
                )
                # Get current state and merge the status matrix properly using controller utility
                # current_global_state = get_current_state()
                current_agent_status_matrix = previous_state.get("agent_status_matrix", {})
                
                # Use controller utility to merge properly
                merged_status_matrix = controller.merge_status_matrix(
                    current_agent_status_matrix,
                    updated_status_matrix
                )
                
                update_current_state({
                    "active_agent": target_agent,
                    "agent_status_matrix": merged_status_matrix,
                    "pending_dependencies": updated_pending_deps,
                    "dependency_graph": updated_dep_graph,
                    "handoff_queue": [
                        *previous_state.get("handoff_queue", []),
                        dependency_request
                    ]
                })

                return Command(
                    goto=target_agent,
                    update={
                        "messages": [*state.get("messages", []), tool_message_with_handoff],
                        "active_agent": target_agent,
                        "agent_status_matrix": updated_status_matrix,
                        "pending_dependencies": updated_pending_deps,
                        "dependency_graph": updated_dep_graph,
                        "agent_workspaces": {
                            **previous_state.get("agent_workspaces", {}),
                            target_agent: {
                                **previous_state.get("agent_workspaces", {}).get(target_agent, {}),
                                "current_task": dependency_request,
                                "context": target_context
                            }
                        },
                        "handoff_queue": [
                            *previous_state.get("handoff_queue", []),
                            dependency_request
                        ]
                    },
                    graph=Command.PARENT
                )
                
            except Exception as e:
                self.logger.log_structured(
                    level="ERROR",
                    message="Dependency handoff failed",
                    extra={
                        "source_agent": active_agent,
                        "target_agent": target_agent,
                        "dependency_type": dependency_type.value,
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                )
                # Create the fallback ToolMessage for the command
                fallback_tool_message = ToolMessage(
                    content=f"Handoff failed, falling back to resource_configuration_agent",
                    name=f"handoff_to_{target_agent}_{dependency_type.value}",
                    tool_call_id=tool_call_id,
                    metadata={
                        "handoff_destination": "resource_configuration_agent",
                        "command_type": "goto",
                        "error": "fallback",
                        "state_updates": {"active_agent": "resource_configuration_agent"}
                    }
                )
                
                fallback_human_message = HumanMessage(
                    content="Please continue with resource configuration due to handoff failure"
                )
                return Command(
                    goto="resource_configuration_agent",
                    update={
                        "messages": [*previous_state.get("messages", []), fallback_tool_message, fallback_human_message],
                        "active_agent": "resource_configuration_agent",
                        "agent_status_matrix": {
                            **previous_state.get("agent_status_matrix", {}),
                            "resource_configuration_agent": GeneratorAgentStatus.ACTIVE
                        },
                    },
                    graph=Command.PARENT
                )
        dependency_handoff_tool.metadata = {METADATA_KEY_HANDOFF_DESTINATION: target_agent}
        return dependency_handoff_tool
    
    def create_target_agent_context(
        self, 
        state: GeneratorSwarmState, 
        dependency_request: Dict[str, Any], 
        target_agent: str,
        transformed_dependency_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create filtered, relevant context for target agent"""
        
        context = {
            "task": dependency_request,
            "relevant_artifacts": {},
            "execution_plan_excerpt": {},
            "cross_agent_data": {}
        }
        
        # Filter relevant artifacts based on agent type and dependency
        if target_agent == "variable_definition_agent":
            # Use transformed data if available, otherwise fall back to original
            dependency_data = transformed_dependency_data or dependency_request.get("dependency_data", {})
            context["relevant_artifacts"] = {
                "resources_needing_variables": dependency_data.get("resources", []),
                "variable_requirements": dependency_data.get("variable_specs", [])
            }
        elif target_agent == "data_source_agent":
            context["relevant_artifacts"] = {
                "external_references": dependency_request["dependency_data"].get("external_refs", []),
                "lookup_requirements": dependency_request["dependency_data"].get("lookups", [])
            }
        elif target_agent == "local_values_agent":
            context["relevant_artifacts"] = {
                "computation_requirements": dependency_request["dependency_data"].get("computations", []),
                "expression_specs": dependency_request["dependency_data"].get("expressions", [])
            }
        
        # Add cross-agent data that might be relevant
        context["cross_agent_data"] = {
            agent: workspace.get("generated_resources", []) + 
                   workspace.get("generated_variables", []) +
                   workspace.get("generated_data_sources", []) +
                   workspace.get("generated_locals", [])
            for agent, workspace in state["agent_workspaces"].items()
            if agent != target_agent
        }
        
        return context

    def create_completion_handoff_tool(self, agent_name: str):
        """Create a completion handoff tool for a specific agent"""
        return create_completion_handoff_tool(agent_name)

    def _parse_tool_message_content(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse the content of a tool message with robust error handling"""
        try:
            # First attempt: Direct JSON parsing
            parsed_data = json.loads(content)
            if isinstance(parsed_data, dict):
                self.logger.log_structured(
                    level="INFO",
                    message="Successfully parsed tool message content",
                    extra={
                        "data_keys": list(parsed_data.keys()) if parsed_data else []
                    }
                )
                return parsed_data
            else:
                self.logger.log_structured(
                    level="ERROR",
                    message="Failed to parse tool message content - not a dictionary",
                    extra={
                        "content_type": type(content).__name__,
                        "parsed_type": type(parsed_data).__name__
                    }
                )
                return None
        except (json.JSONDecodeError, TypeError) as json_error:
            self.logger.log_structured(
                level="WARNING",
                message="JSON parsing failed, attempting content cleaning",
                extra={
                    "json_error": str(json_error),
                    "content_length": len(content)
                }
            )
            
            try:
                # Second attempt: Clean and parse JSON
                cleaned_content = self._clean_json_content(content)
                parsed_data = json.loads(cleaned_content)
                if isinstance(parsed_data, dict):
                    self.logger.log_structured(
                        level="INFO",
                        message="Successfully parsed tool message content after cleaning",
                        extra={
                            "data_keys": list(parsed_data.keys()) if parsed_data else []
                        }
                    )
                    return parsed_data
            except (json.JSONDecodeError, TypeError) as clean_error:
                self.logger.log_structured(
                    level="WARNING",
                    message="Cleaned JSON parsing failed, attempting AST parsing",
                    extra={
                        "clean_error": str(clean_error)
                    }
                )
            
            try:
                # Third attempt: AST literal eval
                parsed_data = ast.literal_eval(content)
                if isinstance(parsed_data, dict):
                    self.logger.log_structured(
                        level="INFO",
                        message="Successfully parsed tool message content using ast.literal_eval",
                        extra={
                            "data_keys": list(parsed_data.keys()) if parsed_data else []
                        }
                    )
                    return parsed_data
                else:
                    self.logger.log_structured(
                        level="ERROR",
                        message="AST parsing failed - not a dictionary",
                        extra={
                            "content_type": type(content).__name__,
                            "parsed_type": type(parsed_data).__name__
                        }
                    )
                    return None
            except (ValueError, SyntaxError) as ast_error:
                self.logger.log_structured(
                    level="ERROR",
                    message="All parsing methods failed",
                    extra={
                        "content_type": type(content).__name__,
                        "json_error": str(json_error),
                        "ast_error": str(ast_error),
                        "content_preview": content[:200] + "..." if len(content) > 200 else content
                    }
                )
                return None
    
    def _clean_json_content(self, content: str) -> str:
        """Clean JSON content to handle common parsing issues"""
        try:
            # Remove any leading/trailing whitespace
            content = content.strip()
            
            # Handle triple-quoted strings that might be causing issues
            if content.startswith('"""') and content.endswith('"""'):
                content = content[3:-3]
            
            # Very targeted approach: only fix the specific hcl_block issue
            # Look for the exact pattern that's causing problems
            import re
            
            # Find hcl_block lines that contain unescaped quotes
            # Pattern: "hcl_block": "resource "aws_vpc" "main" {"
            lines = content.split('\n')
            fixed_lines = []
            
            for line in lines:
                if '"hcl_block":' in line and 'resource "' in line:
                    # This is a problematic hcl_block line - fix it manually
                    # Find the start of the hcl_block value
                    start_idx = line.find('"hcl_block":') + len('"hcl_block":')
                    # Skip whitespace
                    while start_idx < len(line) and line[start_idx] in ' \t':
                        start_idx += 1
                    # Find the opening quote
                    if start_idx < len(line) and line[start_idx] == '"':
                        start_idx += 1
                        # Find the end of the line (assuming it ends with a quote)
                        end_idx = line.rfind('"')
                        if end_idx > start_idx:
                            # Extract the hcl content
                            hcl_content = line[start_idx:end_idx]
                            # Escape all unescaped quotes
                            fixed_hcl_content = hcl_content.replace('"', '\\"')
                            # Reconstruct the line
                            fixed_line = line[:start_idx] + fixed_hcl_content + line[end_idx:]
                            fixed_lines.append(fixed_line)
                        else:
                            fixed_lines.append(line)
                    else:
                        fixed_lines.append(line)
                else:
                    fixed_lines.append(line)
            
            return '\n'.join(fixed_lines)
            
        except Exception as e:
            self.logger.log_structured(
                level="WARNING",
                message="Failed to clean JSON content",
                extra={
                    "error": str(e),
                    "content_length": len(content)
                }
            )
            return content
            

## Completion and Resolution Handoff

def create_completion_handoff_tool(source_agent: str):
    """Tool for agents to signal completion and resolve dependencies
    
    This generic completion tool is used by all agents in the swarm network to:
    - Signal completion of their specific task
    - Pass generated artifacts and results as completion_data
    
    For completion_data parameter:
    - Resource Configuration Agent: Pass generated_resources list
    - Variable Definition Agent: Pass terraform_variables list  
    - Data Source Agent: Pass generated_data_sources list
    - Local Values Agent: Pass generated_locals list
    - Output Definition Agent: Pass generated_outputs list
    - Terraform Readme Agent: Pass readme_content list
    - Terraform Backend Agent: Pass backend_config list

    """
    
    logger = AgentLogger("GeneratorStageHandoffManager")
    
    @tool(f"{source_agent}_complete_task", description=f"Signal completion of {source_agent}'s task")
    def completion_handoff_tool(
        completion_data: Annotated[Dict[str, Any], "Generated artifacts and results - pass the appropriate generated list from your tool response (e.g., terraform_variables, generated_resources, generated_data_sources, etc.)"],
        state: Annotated[Any, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId]
    ) -> Command:
        try:
            logger.log_structured(
                level="INFO",
                message="Processing task completion",
                extra={
                    "source_agent": source_agent,
                    "completion_data_keys": list(completion_data.keys()) if completion_data else []
                }
            )
            global_state = get_current_state()
            # Update agent status to completed (defensive handling for missing keys)
            current_status_matrix = global_state.get("agent_status_matrix", {})
            
            
            # Use controller utility to ensure complete status matrix and merge properly
            controller = GeneratorStageController()
            updated_status_matrix = controller.merge_status_matrix(
                current_status_matrix, 
                {source_agent: GeneratorAgentStatus.COMPLETED}
            )
            
            # CRITICAL FIX: Update the global state immediately with the complete status matrix
            # Get current state and merge the status matrix properly
            current_global_state = get_current_state()
            current_agent_status_matrix = current_global_state.get("agent_status_matrix", {})
            
            # Use controller utility to merge with global state
            merged_status_matrix = controller.merge_status_matrix(
                current_agent_status_matrix,
                updated_status_matrix
            )
            
            update_current_state({
                "agent_status_matrix": merged_status_matrix
            })
            
            # Move resolved dependencies from pending to resolved (defensive handling)
            current_resolved_deps = global_state.get("resolved_dependencies", {})
            updated_resolved_deps = {
                **current_resolved_deps,
                source_agent: [
                    *current_resolved_deps.get(source_agent, [])
                ]
            }
            
            # Remove resolved dependencies from pending (defensive handling)
            current_pending_deps = global_state.get("pending_dependencies", {})
            updated_pending_deps = {}
            for agent, deps in current_pending_deps.items():
                remaining_deps = [
                    dep for dep in deps 
                    if dep["id"] not in [resolved_dep["id"] for resolved_dep in updated_resolved_deps.get(agent, [])]
                ]
                updated_pending_deps[agent] = remaining_deps
            
            # Update agent workspace with completion data (defensive handling)
            current_workspaces = global_state.get("agent_workspaces", {})
            updated_workspaces = {
                **current_workspaces,
                source_agent: {
                    **current_workspaces.get(source_agent, {}),
                    **completion_data,
                    "status": "completed",
                    "completion_timestamp": datetime.datetime.now().isoformat()
                }
            }
            
            # Update progress (defensive handling)
            current_progress = global_state.get("planning_progress", {})
            updated_progress = {
                **current_progress,
                source_agent: 1.0
            }
            
            # Determine next agent based on recommendations and dependencies
            controller = GeneratorStageController()
            # Extract only the required fields to avoid conflicts with old agent_status_matrix
            planning_progress = global_state.get("planning_progress", {})
            dependency_graph = global_state.get("dependency_graph", {})
            active_agent = global_state.get("active_agent")
            agent_waiting_times = global_state.get("agent_waiting_times", {})
            
            
            next_agent = controller.determine_next_active_agent_from_params(
                agent_status_matrix=updated_status_matrix,  # Use updated status matrix
                dependency_graph=dependency_graph,
                active_agent=active_agent,
                agent_waiting_times=agent_waiting_times
            )
            
            # Check if stage is complete
            stage_complete = controller.check_stage_completion_conditions_from_params(
                agent_status_matrix=updated_status_matrix,  # Use updated status matrix
                planning_progress=updated_progress,
                pending_dependencies=updated_pending_deps
            )

            tool_message = ToolMessage(
                content=f"Completion handoff to {source_agent}",
                name=f"handoff_to_{source_agent}_complete_task",
                tool_call_id=tool_call_id
            )

            if stage_complete:
                logger.log_structured(
                    level="INFO",
                    message="Planning stage completed",
                    extra={
                        "source_agent": source_agent,
                        "next_action": "check_planning_stage_completion",
                        "resolved_deps_count": len(updated_resolved_deps)
                    }
                )
                
                # Create ToolMessage and return Command
                completion_tool_message = ToolMessage(
                    content=f"Task completion for {source_agent} - Planning stage complete",
                    name=f"{source_agent}_complete_task",
                    tool_call_id=tool_call_id,
                    metadata={
                        "completion_type": "stage_complete",
                        "next_destination": "check_planning_stage_completion"
                    }
                )
                
                return Command(
                    goto="supervisor",
                    update={
                        "messages": [*state.get("messages", []), completion_tool_message],
                        "agent_status_matrix": updated_status_matrix,
                        "resolved_dependencies": updated_resolved_deps,
                        "pending_dependencies": updated_pending_deps,
                        "agent_workspaces": updated_workspaces,
                        "planning_progress": updated_progress,
                        "stage_status": "planning_complete"
                    },
                    graph=Command.PARENT
                )
            else:
                logger.log_structured(
                    level="INFO",
                    message="Task completed, transitioning to next agent",
                    extra={
                        "source_agent": source_agent,
                        "next_agent": next_agent,
                        "resolved_deps_count": len(updated_resolved_deps),
                        "stage_complete": False
                    }
                )
                
                # Create ToolMessage and return Command
                completion_tool_message = ToolMessage(
                    content=f"Task completion for {source_agent} - Transitioning to {next_agent}. Generated artifacts: {list(completion_data.keys()) if completion_data else 'None'}. Resolved dependencies: {len(updated_resolved_deps.get(source_agent, []))} items.",
                    name=f"{source_agent}_complete_task",
                    tool_call_id=tool_call_id,
                    metadata={
                        "completion_type": "agent_transition",
                        "next_destination": next_agent,
                        "source_agent": source_agent,
                        "completion_data_keys_of_source_agent": list(completion_data.keys()) if completion_data else [],
                        "resolved_dependencies_count_of_source_agent": len(updated_resolved_deps.get(source_agent, [])),
                        "stage_complete": stage_complete
                    }
                )

                # Simply update the next agent to ACTIVE in the already-merged status matrix
                final_status_matrix = {
                    **merged_status_matrix,  # Use the already-merged matrix
                    next_agent: GeneratorAgentStatus.ACTIVE
                }
                
                update_current_state({
                    "active_agent": next_agent,
                    "agent_status_matrix": final_status_matrix,
                    "resolved_dependencies": updated_resolved_deps,
                    "pending_dependencies": updated_pending_deps,
                    "agent_workspaces": updated_workspaces,
                    "planning_progress": updated_progress
                })  

                return Command(
                    goto=next_agent,
                    update={
                        "messages": [*state.get("messages", []), completion_tool_message],
                        "active_agent": next_agent,
                        "agent_status_matrix": {
                            **updated_status_matrix,
                            next_agent: GeneratorAgentStatus.ACTIVE
                        },
                        "resolved_dependencies": updated_resolved_deps,
                        "pending_dependencies": updated_pending_deps,
                        "agent_workspaces": updated_workspaces,
                        "planning_progress": updated_progress
                    },
                    graph=Command.PARENT
                )
                
        except Exception as e:
            logger.log_structured(
                level="ERROR",
                message="Task completion handoff failed",
                extra={
                    "source_agent": source_agent,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            # Create error fallback ToolMessage and return Command
            error_tool_message = ToolMessage(
                content=f"Task completion failed for {source_agent}, falling back to resource_configuration_agent",
                name=f"{source_agent}_complete_task",
                tool_call_id=tool_call_id,
                metadata={
                    "completion_type": "error_fallback",
                    "next_destination": "resource_configuration_agent",
                    "error": str(e)
                }
            )
            
            return Command(
                goto="resource_configuration_agent",
                update={
                    "messages": [*state.get("messages", []), error_tool_message],
                    "active_agent": "resource_configuration_agent",
                    "agent_status_matrix": {
                        **state.get("agent_status_matrix", {}),
                        "resource_configuration_agent": GeneratorAgentStatus.ACTIVE
                    }
                },
                graph=Command.PARENT
            )
    
    # Set metadata for completion tool (may be used by swarm system)
    completion_handoff_tool.metadata = {"completion_tool": True, "source_agent": source_agent}
    return completion_handoff_tool

