from typing import Annotated, Dict, Any, List
from langchain_core.tools import tool
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from langgraph.checkpoint.base import BaseCheckpointSaver
from .generator_state import GeneratorSwarmState, GeneratorAgentStatus
from aws_orchestrator_agent.utils.logger import AgentLogger
import datetime

class GeneratorStageCheckpointManager:
    """Manages checkpointing for Planning Stage with comprehensive metadata collection"""
    
    def __init__(self, checkpointer: BaseCheckpointSaver):
        self.checkpointer = checkpointer
        self.checkpoint_frequency = 30  # seconds
        self.logger = AgentLogger("GeneratorStageCheckpointManager")
    
    def create_checkpoint(self, state: GeneratorSwarmState, event_type: str) -> Dict[str, Any]:
        """Create comprehensive checkpoint with recovery metadata"""
        try:
            checkpoint_id = f"planning_{int(datetime.datetime.now().timestamp())}"
            
            self.logger.log_structured(
                level="DEBUG",
                message="Creating checkpoint",
                extra={
                    "checkpoint_id": checkpoint_id,
                    "event_type": event_type,
                    "active_agent": state.get("active_agent", "unknown")
                }
            )
            
            checkpoint_data = {
                "checkpoint_id": checkpoint_id,
                "event_type": event_type,  # "handoff", "completion", "error", "manual"
                "timestamp": datetime.datetime.now().isoformat(),
                
                # Core state
                "state_snapshot": state,
                
                # Recovery metadata
                "recovery_metadata": {
                    "active_dependencies": self.extract_active_dependencies(state),
                    "agent_execution_stack": self.build_execution_stack(state),
                    "pending_operations": self.identify_pending_operations(state),
                    "rollback_points": self.identify_rollback_points(state)
                },
                
                # Performance metadata
                "performance_data": {
                    "stage_duration": self.calculate_stage_duration(state),
                    "agent_performance": self.calculate_agent_performance(state)
                }
            }
            
            # Store checkpoint (simplified for InMemorySaver compatibility)
            # Note: InMemorySaver doesn't have put_checkpoint method
            # This is a simplified checkpoint storage for testing
            if hasattr(self.checkpointer, 'put_checkpoint'):
                self.checkpointer.put_checkpoint(
                    config={"checkpoint_id": checkpoint_data["checkpoint_id"]},
                    checkpoint=checkpoint_data
                )
            else:
                # For InMemorySaver, we'll just log the checkpoint creation
                # In a real implementation, you'd use the proper checkpoint API
                self.logger.log_structured(
                    level="DEBUG",
                    message="Checkpoint data prepared (InMemorySaver compatibility)",
                    extra={"checkpoint_id": checkpoint_data["checkpoint_id"]}
                )
            
            self.logger.log_structured(
                level="INFO",
                message="Checkpoint created successfully",
                extra={
                    "checkpoint_id": checkpoint_id,
                    "event_type": event_type,
                    "checkpoint_size": len(str(checkpoint_data))
                }
            )
            
            return checkpoint_data
            
        except Exception as e:
            self.logger.log_structured(
                level="ERROR",
                message="Failed to create checkpoint",
                extra={
                    "event_type": event_type,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            raise
    
    def test_checkpoint_current_state(self, event_type: str, state: GeneratorSwarmState) -> bool:
        """Test version of checkpoint_current_state without @tool decorator"""
        try:
            checkpoint_data = self.create_checkpoint(state, event_type)
            return True
        except Exception as e:
            self.logger.log_structured(
                level="ERROR",
                message="Error in test_checkpoint_current_state",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            return False
    

    @tool("create_planning_checkpoint")
    def checkpoint_current_state(
        self,
        event_type: Annotated[str, "Type of checkpoint event"],
        state: Annotated[GeneratorSwarmState, InjectedState]
    ) -> Command:
        """Tool for agents to create checkpoints at critical points"""
        try:
            checkpoint_data = self.create_checkpoint(state, event_type)
            
            self.logger.log_structured(
                level="DEBUG",
                message="Checkpoint tool executed",
                extra={
                    "event_type": event_type,
                    "checkpoint_id": checkpoint_data["checkpoint_id"],
                    "active_agent": state.get("active_agent", "unknown")
                }
            )
            
            return Command(
                goto=state.get("active_agent", "resource_configuration_agent"),
                update={
                    "checkpoint_metadata": {
                        **state.get("checkpoint_metadata", {}),
                        "last_checkpoint": checkpoint_data["checkpoint_id"],
                        "checkpoint_timestamp": checkpoint_data["timestamp"]
                    }
                },
                graph=Command.PARENT
            )
            
        except Exception as e:
            self.logger.log_structured(
                level="ERROR",
                message="Checkpoint tool failed",
                extra={
                    "event_type": event_type,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            # Return safe fallback command
            return Command(
                goto="resource_configuration_agent",
                update={"active_agent": "resource_configuration_agent"},
                graph=Command.PARENT
            )
    
    # Helper methods for checkpoint metadata extraction
    def extract_active_dependencies(self, state: GeneratorSwarmState) -> Dict[str, List[str]]:
        """Extract currently active dependencies from state"""
        try:
            active_deps = {}
            for agent, deps in state.get("pending_dependencies", {}).items():
                if deps:  # Only include agents with pending dependencies
                    active_deps[agent] = [dep.get("type", "unknown") for dep in deps]
            return active_deps
        except Exception as e:
            self.logger.log_structured(
                level="ERROR",
                message="Failed to extract active dependencies",
                extra={"error": str(e)}
            )
            return {}
    
    def build_execution_stack(self, state: GeneratorSwarmState) -> List[str]:
        """Build execution stack from current state"""
        try:
            execution_stack = []
            for agent, status in state.get("agent_status_matrix", {}).items():
                if status == GeneratorAgentStatus.ACTIVE:
                    execution_stack.append(agent)
            return execution_stack
        except Exception as e:
            self.logger.log_structured(
                level="ERROR",
                message="Failed to build execution stack",
                extra={"error": str(e)}
            )
            return []
    
    def identify_pending_operations(self, state: GeneratorSwarmState) -> Dict[str, List[str]]:
        """Identify pending operations for each agent"""
        try:
            pending_ops = {}
            for agent, workspace in state.get("agent_workspaces", {}).items():
                ops = []
                if workspace.get("pending_variable_requests"):
                    ops.append("variable_requests")
                if workspace.get("pending_data_source_requests"):
                    ops.append("data_source_requests")
                if workspace.get("completion_checklist"):
                    ops.append("completion_tasks")
                if ops:
                    pending_ops[agent] = ops
            return pending_ops
        except Exception as e:
            self.logger.log_structured(
                level="ERROR",
                message="Failed to identify pending operations",
                extra={"error": str(e)}
            )
            return {}
    
    def identify_rollback_points(self, state: GeneratorSwarmState) -> List[Dict[str, Any]]:
        """Identify potential rollback points in the execution"""
        try:
            rollback_points = []
            
            # Add checkpoint metadata as rollback point
            if state.get("checkpoint_metadata"):
                rollback_points.append({
                    "type": "checkpoint",
                    "checkpoint_id": state["checkpoint_metadata"].get("last_checkpoint"),
                    "timestamp": state["checkpoint_metadata"].get("checkpoint_timestamp")
                })
            
            # Add completed agents as rollback points
            for agent, status in state.get("agent_status_matrix", {}).items():
                if status == GeneratorAgentStatus.COMPLETED:
                    rollback_points.append({
                        "type": "agent_completion",
                        "agent": agent,
                        "status": status.value
                    })
            
            return rollback_points
        except Exception as e:
            self.logger.log_structured(
                level="ERROR",
                message="Failed to identify rollback points",
                extra={"error": str(e)}
            )
            return []
    
    def calculate_stage_duration(self, state: GeneratorSwarmState) -> float:
        """Calculate total stage duration"""
        try:
            start_time = state.get("stage_start_time")
            if start_time:
                if isinstance(start_time, str):
                    start_dt = datetime.datetime.fromisoformat(start_time)
                else:
                    start_dt = start_time
                return (datetime.datetime.now() - start_dt).total_seconds()
            return 0.0
        except Exception as e:
            self.logger.log_structured(
                level="ERROR",
                message="Failed to calculate stage duration",
                extra={"error": str(e)}
            )
            return 0.0
    
    def calculate_agent_performance(self, state: GeneratorSwarmState) -> Dict[str, Dict[str, Any]]:
        """Calculate performance metrics for each agent"""
        try:
            performance = {}
            for agent, workspace in state.get("agent_workspaces", {}).items():
                agent_perf = {
                    "retry_count": workspace.get("retry_count", 0),
                    "last_failure": workspace.get("last_failure"),
                    "completion_progress": len(workspace.get("completion_checklist", [])),
                    "status": state.get("agent_status_matrix", {}).get(agent, "unknown")
                }
                performance[agent] = agent_perf
            return performance
        except Exception as e:
            self.logger.log_structured(
                level="ERROR",
                message="Failed to calculate agent performance",
                extra={"error": str(e)}
            )
            return {}
    
