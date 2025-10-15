from typing import Annotated
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from langchain_core.tools import tool
from .generator_state import GeneratorSwarmState, GeneratorAgentStatus
from aws_orchestrator_agent.utils.logger import AgentLogger    


class GeneratorStageController:
    """Orchestrates the Planning Stage with parallel execution and dependency management"""
    
    def __init__(self):
        self.completion_threshold = 0.95  # 95% completion required
        self.max_dependency_wait_time = 300  # 5 minutes max wait
        self.parallel_execution_slots = 4  # All 4 agents can run in parallel
        self.logger = AgentLogger("GeneratorStageController")
        
    def initialize_planning_stage(self, state: GeneratorSwarmState) -> Command:
        """Initialize planning stage with Resource Configuration Agent as entry point"""
        return Command(
            goto="resource_configuration_agent",
            update={
                "stage_status": "planning_active",
                "agent_status_matrix": {
                    **state["agent_status_matrix"],
                    "resource_configuration_agent": GeneratorAgentStatus.ACTIVE
                },
                "planning_progress": {
                    "resource_configuration_agent": 0.0,
                    "variable_definition_agent": 0.0,
                    "data_source_agent": 0.0,
                    "local_values_agent": 0.0,
                    "output_definition_agent": 0.0,
                    "terraform_backend_generator": 0.0,
                    "terraform_readme_generator": 0.0
                }
            },
            graph=Command.PARENT
        )
    
    def calculate_stage_completion(self, state: GeneratorSwarmState) -> float:
        """Calculate overall planning stage completion percentage"""
        progress_values = list(state["planning_progress"].values())
        if not progress_values:
            return 0.0
            
        # Weighted completion based on criticality
        weights = {
            "resource_configuration_agent": 0.30,  # Core infrastructure - highest
            "variable_definition_agent": 0.25,     # Core configuration - high
            "data_source_agent": 0.20,            # Data dependencies - medium-high
            "local_values_agent": 0.15,           # Computed values - medium
            "output_definition_agent": 0.10,      # Output definitions - low
            "terraform_backend_generator": 0.00,  # Backend configuration - supporting
            "terraform_readme_generator": 0.00    # Documentation - supporting
        }
        
        weighted_sum = sum(
            weights.get(agent, 0.25) * progress 
            for agent, progress in state["planning_progress"].items()
        )
        
        return min(weighted_sum, 1.0)
    
    def calculate_stage_completion_from_params(self, planning_progress: dict) -> float:
        """Parameterized version of calculate_stage_completion"""
        progress_values = list(planning_progress.values())
        if not progress_values:
            return 0.0
            
        # Weighted completion based on criticality
        weights = {
            "resource_configuration_agent": 0.30,  # Core infrastructure - highest
            "variable_definition_agent": 0.25,     # Core configuration - high
            "data_source_agent": 0.20,            # Data dependencies - medium-high
            "local_values_agent": 0.15,           # Computed values - medium
            "output_definition_agent": 0.10,      # Output definitions - low
            "terraform_backend_generator": 0.00,  # Backend configuration - supporting
            "terraform_readme_generator": 0.00    # Documentation - supporting
        }
        
        weighted_sum = sum(
            weights.get(agent, 0.25) * progress 
            for agent, progress in planning_progress.items()
        )
        
        return min(weighted_sum, 1.0)
    
    def check_stage_completion_conditions(self, state: GeneratorSwarmState) -> bool:
        """Comprehensive stage completion check"""
        # Check 1: All agents must be completed
        required_agents = [
            "resource_configuration_agent",
            "variable_definition_agent", 
            "data_source_agent",
            "local_values_agent",
            "output_definition_agent",
            "terraform_backend_generator",
            "terraform_readme_generator"
        ]
        
        agents_completed = all(
            state["agent_status_matrix"].get(agent) == GeneratorAgentStatus.COMPLETED
            for agent in required_agents
        )
        
        # Check 2: No pending dependencies
        no_pending_deps = all(
            len(deps) == 0 for deps in state["pending_dependencies"].values()
        )
        
        # Check 3: Overall completion threshold met
        completion_threshold_met = self.calculate_stage_completion(state) >= self.completion_threshold
        
        # Check 4: No agents in error state (unless recovered)
        no_error_agents = all(
            status != GeneratorAgentStatus.ERROR 
            for status in state["agent_status_matrix"].values()
        )
        
        return agents_completed and no_pending_deps and completion_threshold_met and no_error_agents
    
    def check_stage_completion_conditions_from_params(
        self, 
        agent_status_matrix: dict,
        planning_progress: dict,
        pending_dependencies: dict
    ) -> bool:
        """Parameterized version of check_stage_completion_conditions"""
        # Check 1: All agents must be completed
        required_agents = [
            "resource_configuration_agent",
            "variable_definition_agent", 
            "data_source_agent",
            "local_values_agent",
            "output_definition_agent",
            "terraform_backend_generator",
            "terraform_readme_generator"
        ]
        
        agents_completed = all(
            agent_status_matrix.get(agent) == GeneratorAgentStatus.COMPLETED
            for agent in required_agents
        )
        
        # Check 2: No pending dependencies
        no_pending_deps = all(
            len(deps) == 0 for deps in pending_dependencies.values()
        )
        
        # Check 3: Overall completion threshold met
        completion_threshold_met = self.calculate_stage_completion_from_params(planning_progress) >= self.completion_threshold
        
        # Check 4: No agents in error state (unless recovered)
        no_error_agents = all(
            status != GeneratorAgentStatus.ERROR 
            for status in agent_status_matrix.values()
        )
        
        return agents_completed and no_pending_deps and completion_threshold_met and no_error_agents

    @tool("check_planning_stage_completion")
    def check_and_transition_stage(
        self, 
        state: Annotated[GeneratorSwarmState, InjectedState]
    ) -> Command:
        """Check completion and transition to Enhancement stage if ready"""
        try:
            # Validate state integrity first
            if not self.validate_state_integrity(state):
                self.logger.log_structured(
                    level="ERROR",
                    message="State integrity validation failed, defaulting to resource agent",
                    extra={"fallback_agent": "resource_configuration_agent"}
                )
                return Command(
                    goto="resource_configuration_agent",
                    update={"active_agent": "resource_configuration_agent"},
                    graph=Command.PARENT
                )
            
            if self.check_stage_completion_conditions(state):
                # Prepare transition to Enhancement Stage
                self.logger.log_structured(
                    level="INFO",
                    message="Planning stage completed, transitioning to Enhancement stage",
                    extra={
                        "stage_completion": self.calculate_stage_completion(state),
                        "next_stage": "enhancement",
                        "next_agent": "security_compliance_agent"
                    }
                )
                return Command(
                    goto="security_compliance_agent",  # First agent in Enhancement stage
                    update={
                        "stage_status": "planning_complete",
                        "current_stage": "enhancement",
                        "stage_progress": {
                            **state.get("stage_progress", {}),
                            "planning": 1.0,
                            "enhancement": 0.0
                        }
                    },
                    graph=Command.PARENT
                )
            else:
                # Determine next action based on current state
                next_agent = self.determine_next_active_agent(state)
                self.logger.log_structured(
                    level="DEBUG",
                    message="Planning stage not complete, selecting next agent",
                    extra={
                        "selected_agent": next_agent,
                        "stage_completion": self.calculate_stage_completion(state)
                    }
                )
                return Command(
                    goto=next_agent,
                    update={
                        "active_agent": next_agent
                    },
                    graph=Command.PARENT
                )
                
        except Exception as e:
            self.logger.log_structured(
                level="ERROR",
                message="Error in check_and_transition_stage",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "fallback_agent": "resource_configuration_agent"
                }
            )
            # Fallback to safe default
            return Command(
                goto="resource_configuration_agent",
                update={"active_agent": "resource_configuration_agent"},
                graph=Command.PARENT
            )
    
    def determine_next_active_agent(self, state: GeneratorSwarmState) -> str:
        """Intelligent agent selection based on dependencies and priorities"""
        try:
            # Priority 1: Agents with resolved dependencies
            ready_agents = []
            
            # Define all required agents
            required_agents = [
                "resource_configuration_agent",
                "variable_definition_agent", 
                "data_source_agent",
                "local_values_agent",
                "output_definition_agent",
                "terraform_backend_generator",
                "terraform_readme_generator"
            ]
            
            for agent in required_agents:
                # Get status from matrix, default to INACTIVE if not present
                status = state["agent_status_matrix"].get(agent, GeneratorAgentStatus.INACTIVE)
                
                if status in [GeneratorAgentStatus.INACTIVE, GeneratorAgentStatus.WAITING]:
                    dependencies_met = self.check_agent_dependencies_met(agent, state)
                    if dependencies_met:
                        priority = self.get_agent_priority(agent, state)
                        ready_agents.append((agent, priority))
            
            if ready_agents:
                # Return highest priority agent
                selected_agent = max(ready_agents, key=lambda x: x[1])[0]
                self.logger.log_structured(
                    level="DEBUG",
                    message="Selected next agent from ready agents",
                    extra={
                        "selected_agent": selected_agent,
                        "ready_agents_count": len(ready_agents),
                        "ready_agents": [agent for agent, _ in ready_agents]
                    }
                )
                return selected_agent
            
            # Priority 2: Currently active agent (continue execution)
            current_active = state.get("active_agent")
            if (current_active and 
                state["agent_status_matrix"].get(current_active) == GeneratorAgentStatus.ACTIVE):
                self.logger.log_structured(
                    level="DEBUG",
                    message="Continuing with current active agent",
                    extra={"current_agent": current_active}
                )
                return current_active
                
            # Priority 3: Default to Resource Configuration Agent
            self.logger.log_structured(
                level="DEBUG",
                message="Defaulting to resource_configuration_agent",
                extra={"reason": "no_ready_agents_or_active_agent"}
            )
            return "resource_configuration_agent"
            
        except Exception as e:
            self.logger.log_structured(
                level="ERROR",
                message="Error in determine_next_active_agent",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "fallback_agent": "resource_configuration_agent"
                }
            )
            # Fallback to safe default
            return "resource_configuration_agent"
    
    def determine_next_active_agent_from_params(
        self, 
        agent_status_matrix: dict,
        dependency_graph: dict,
        active_agent: str = None,
        agent_waiting_times: dict = None
    ) -> str:
        """Parameterized version of determine_next_active_agent for handoff tools"""
        try:
            if agent_waiting_times is None:
                agent_waiting_times = {}
                
            # Priority 1: Agents with resolved dependencies
            ready_agents = []
            
            # Define all required agents
            required_agents = [
                "resource_configuration_agent",
                "variable_definition_agent", 
                "data_source_agent",
                "local_values_agent",
                "output_definition_agent",
                "terraform_backend_generator",
                "terraform_readme_generator"
            ]
            
            for agent in required_agents:
                # Get status from matrix, default to INACTIVE if not present
                status = agent_status_matrix.get(agent, GeneratorAgentStatus.INACTIVE)
                
                
                if status in [GeneratorAgentStatus.INACTIVE, GeneratorAgentStatus.WAITING]:
                    dependencies_met = self.check_agent_dependencies_met_from_params(
                        agent, agent_status_matrix, dependency_graph
                    )
                    if dependencies_met:
                        priority = self.get_agent_priority_from_params(
                            agent, agent_waiting_times
                        )
                        ready_agents.append((agent, priority))
            
            
            if ready_agents:
                # Return highest priority agent
                selected_agent = max(ready_agents, key=lambda x: x[1])[0]
                self.logger.log_structured(
                    level="DEBUG",
                    message="Selected next agent from ready agents (parameterized)",
                    extra={
                        "selected_agent": selected_agent,
                        "ready_agents_count": len(ready_agents),
                        "ready_agents": [agent for agent, _ in ready_agents]
                    }
                )
                return selected_agent
            
            # Priority 2: Currently active agent (continue execution)
            if (active_agent and 
                agent_status_matrix.get(active_agent) == GeneratorAgentStatus.ACTIVE):
                self.logger.log_structured(
                    level="DEBUG",
                    message="Continuing with current active agent (parameterized)",
                    extra={"current_agent": active_agent}
                )
                return active_agent
                
            # Priority 3: Default to Resource Configuration Agent
            self.logger.log_structured(
                level="DEBUG",
                message="Defaulting to resource_configuration_agent (parameterized)",
                extra={"reason": "no_ready_agents_or_active_agent"}
            )
            return "resource_configuration_agent"
            
        except Exception as e:
            self.logger.log_structured(
                level="ERROR",
                message="Error in determine_next_active_agent_from_params",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "fallback_agent": "resource_configuration_agent"
                }
            )
            # Fallback to safe default
            return "resource_configuration_agent"
    
    def check_agent_dependencies_met(self, agent_name: str, state: GeneratorSwarmState) -> bool:
        """Check if agent's dependencies are satisfied"""
        required_deps = state["dependency_graph"].get(agent_name, set())
        
        for dep_agent in required_deps:
            dep_status = state["agent_status_matrix"].get(dep_agent)
            if dep_status != GeneratorAgentStatus.COMPLETED:
                return False
                
        return True
    
    def check_agent_dependencies_met_from_params(
        self, 
        agent_name: str, 
        agent_status_matrix: dict, 
        dependency_graph: dict
    ) -> bool:
        """Parameterized version of check_agent_dependencies_met"""
        required_deps = dependency_graph.get(agent_name, set())
        
        for dep_agent in required_deps:
            dep_status = agent_status_matrix.get(dep_agent)
            if dep_status != GeneratorAgentStatus.COMPLETED:
                return False
                
        return True
    
    def ensure_complete_status_matrix(self, current_status_matrix: dict) -> dict:
        """Ensure all required agents are in the status matrix with proper statuses"""
        required_agents = [
            "resource_configuration_agent",
            "variable_definition_agent", 
            "data_source_agent",
            "local_values_agent",
            "output_definition_agent",
            "terraform_backend_generator",
            "terraform_readme_generator"
        ]
        
        # Initialize missing agents with INACTIVE status and normalize existing ones
        complete_status_matrix = {}
        for agent in required_agents:
            if agent in current_status_matrix:
                # Normalize string statuses to enum statuses
                status = current_status_matrix[agent]
                if isinstance(status, str):
                    # Convert string to enum
                    if status == 'inactive':
                        complete_status_matrix[agent] = GeneratorAgentStatus.INACTIVE
                    elif status == 'active':
                        complete_status_matrix[agent] = GeneratorAgentStatus.ACTIVE
                    elif status == 'waiting':
                        complete_status_matrix[agent] = GeneratorAgentStatus.WAITING
                    elif status == 'completed':
                        complete_status_matrix[agent] = GeneratorAgentStatus.COMPLETED
                    elif status == 'error':
                        complete_status_matrix[agent] = GeneratorAgentStatus.ERROR
                    else:
                        complete_status_matrix[agent] = GeneratorAgentStatus.INACTIVE
                else:
                    # Already an enum, use as-is
                    complete_status_matrix[agent] = status
            else:
                complete_status_matrix[agent] = GeneratorAgentStatus.INACTIVE
        
        return complete_status_matrix
    
    def merge_status_matrix(self, current_status_matrix: dict, updates: dict) -> dict:
        """Merge status matrix updates while preserving existing statuses"""
        # Ensure complete status matrix first
        complete_matrix = self.ensure_complete_status_matrix(current_status_matrix)
        
        # Merge with updates
        merged_matrix = {
            **complete_matrix,
            **updates
        }
        
        return merged_matrix

    def get_agent_priority(self, agent_name: str, state: GeneratorSwarmState) -> int:
        """Calculate agent priority based on criticality and current state"""
        try:
            # Define priority weights (higher = more important)
            priority_weights = {
                "resource_configuration_agent": 6,  # Highest priority - core infrastructure
                "variable_definition_agent": 5,     # High priority - core configuration
                "data_source_agent": 4,            # Medium-high priority - data dependencies
                "local_values_agent": 3,           # Medium priority - computed values
                "output_definition_agent": 4,      # Medium-high priority - outputs
                "terraform_backend_generator": 1,  # Lowest priority - supporting backend
                "terraform_readme_generator": 1   # Lowest priority - supporting documentation
            }
            
            base_priority = priority_weights.get(agent_name, 0)
            
            # Boost priority if agent has been waiting longer
            # This helps prevent starvation of lower-priority agents
            waiting_time = state.get("agent_waiting_times", {}).get(agent_name, 0)
            time_boost = min(waiting_time // 60, 2)  # Max 2 point boost for waiting 1+ minutes
            
            total_priority = base_priority + time_boost
            self.logger.log_structured(
                level="DEBUG",
                message="Calculated agent priority",
                extra={
                    "agent_name": agent_name,
                    "base_priority": base_priority,
                    "time_boost": time_boost,
                    "total_priority": total_priority,
                    "waiting_time": waiting_time
                }
            )
            
            return total_priority
            
        except Exception as e:
            self.logger.log_structured(
                level="ERROR",
                message="Error calculating priority for agent",
                extra={
                    "agent_name": agent_name,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "fallback_priority": 1
                }
            )
            # Return default priority for unknown agents
            return 1
    
    def get_agent_priority_from_params(self, agent_name: str, agent_waiting_times: dict) -> int:
        """Parameterized version of get_agent_priority"""
        try:
            # Define priority weights (higher = more important)
            priority_weights = {
                "resource_configuration_agent": 6,  # Highest priority - core infrastructure
                "variable_definition_agent": 5,     # High priority - core configuration
                "data_source_agent": 4,            # Medium-high priority - data dependencies
                "local_values_agent": 3,           # Medium priority - computed values
                "output_definition_agent": 4,      # Medium-high priority - outputs
                "terraform_backend_generator": 1,  # Lowest priority - supporting backend
                "terraform_readme_generator": 1   # Lowest priority - supporting documentation
            }
            
            base_priority = priority_weights.get(agent_name, 0)
            
            # Boost priority if agent has been waiting longer
            # This helps prevent starvation of lower-priority agents
            waiting_time = agent_waiting_times.get(agent_name, 0)
            time_boost = min(waiting_time // 60, 2)  # Max 2 point boost for waiting 1+ minutes
            
            total_priority = base_priority + time_boost
            self.logger.log_structured(
                level="DEBUG",
                message="Calculated agent priority (parameterized)",
                extra={
                    "agent_name": agent_name,
                    "base_priority": base_priority,
                    "time_boost": time_boost,
                    "total_priority": total_priority,
                    "waiting_time": waiting_time
                }
            )
            
            return total_priority
            
        except Exception as e:
            self.logger.log_structured(
                level="ERROR",
                message="Error calculating priority for agent (parameterized)",
                extra={
                    "agent_name": agent_name,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "fallback_priority": 1
                }
            )
            # Return default priority for unknown agents
            return 1
    
    def validate_state_integrity(self, state: GeneratorSwarmState) -> bool:
        """Validate that the state has all required fields and is in a valid condition"""
        try:
            # Check required fields exist
            required_fields = [
                "agent_status_matrix", 
                "planning_progress", 
                "dependency_graph",
                "pending_dependencies"
            ]
            
            for field in required_fields:
                if field not in state:
                    self.logger.log_structured(
                        level="ERROR",
                        message="Missing required state field",
                        extra={"missing_field": field, "required_fields": required_fields}
                    )
                    return False
            
            # Check agent status matrix has all required agents
            required_agents = [
                "resource_configuration_agent",
                "variable_definition_agent", 
                "data_source_agent",
                "local_values_agent",
                "output_definition_agent",
                "terraform_backend_generator",
                "terraform_readme_generator"
            ]
            
            for agent in required_agents:
                if agent not in state["agent_status_matrix"]:
                    self.logger.log_structured(
                        level="ERROR",
                        message="Missing agent in status matrix",
                        extra={"missing_agent": agent, "required_agents": required_agents}
                    )
                    return False
            
            # Check progress tracking has all agents
            for agent in required_agents:
                if agent not in state["planning_progress"]:
                    self.logger.log_structured(
                        level="ERROR",
                        message="Missing agent in progress tracking",
                        extra={"missing_agent": agent, "required_agents": required_agents}
                    )
                    return False
            
            self.logger.log_structured(
                level="DEBUG",
                message="State integrity validation passed",
                extra={
                    "validated_fields": len(required_fields),
                    "validated_agents": len(required_agents)
                }
            )
            return True
            
        except Exception as e:
            self.logger.log_structured(
                level="ERROR",
                message="Error validating state integrity",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            return False
    
    def test_check_and_transition_stage(self, state: GeneratorSwarmState) -> bool:
        """Test version of check_and_transition_stage without @tool decorator"""
        try:
            # Validate state integrity first
            if not self.validate_state_integrity(state):
                self.logger.log_structured(
                    level="ERROR",
                    message="State integrity validation failed, defaulting to resource agent",
                    extra={"fallback_agent": "resource_configuration_agent"}
                )
                return False
            
            return self.check_stage_completion_conditions(state)
                
        except Exception as e:
            self.logger.log_structured(
                level="ERROR",
                message="Error in test_check_and_transition_stage",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "fallback_agent": "resource_configuration_agent"
                }
            )
            return False
