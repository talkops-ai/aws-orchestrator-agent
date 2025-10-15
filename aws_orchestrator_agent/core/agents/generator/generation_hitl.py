from typing import Annotated, Dict, Any
from langchain_core.tools import tool
from langgraph.types import Command
from langgraph.prebuilt import InjectedState
from langchain_core.tools import InjectedToolCallId
from .generator_state import GeneratorSwarmState, GeneratorAgentStatus
from .generator_state_controller import GeneratorStageController
from aws_orchestrator_agent.utils.logger import AgentLogger
import datetime

class GeneratorStageHumanLoop:
    """Manages human-in-the-loop interactions for Generator Stage"""
    
    def __init__(self):
        self.approval_thresholds = {
            "high_cost_resources": 1000,  # USD
            "security_critical_changes": True,
            "cross_region_dependencies": True,
            "experimental_features": True
        }
        self.logger = AgentLogger("GeneratorStageHumanLoop")
    
    def create_approval_checkpoint_tool(self, trigger_condition: str):
        """Create dynamic approval checkpoint tools"""
        
        @tool(f"request_approval_{trigger_condition}")
        def approval_checkpoint_tool(
            state: Annotated[GeneratorSwarmState, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],
            approval_context: Annotated[Dict[str, Any], "Context requiring approval"],
            urgency_level: Annotated[int, "Urgency: 1=low, 5=critical"] = 3,
            timeout_minutes: Annotated[int, "Minutes to wait for approval"] = 30,
        ) -> Command:
            """Request human approval for a specific trigger condition."""
            try:
                requesting_agent = state["active_agent"]
                
                self.logger.log_structured(
                    level="WARNING",
                    message="Human approval requested",
                    extra={
                        "trigger_condition": trigger_condition,
                        "requesting_agent": requesting_agent,
                        "urgency_level": urgency_level,
                        "timeout_minutes": timeout_minutes,
                        "approval_context_keys": list(approval_context.keys()) if approval_context else []
                    }
                )
                
                approval_request = {
                    "id": f"approval_{trigger_condition}_{int(datetime.datetime.now().timestamp())}",
                    "trigger_condition": trigger_condition,
                    "context": approval_context,
                    "urgency_level": urgency_level,
                    "timeout_minutes": timeout_minutes,
                    "requested_by_agent": requesting_agent,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "status": "pending"
                }
            
                self.logger.log_structured(
                    level="INFO",
                    message="Approval request created and queued",
                    extra={
                        "approval_id": approval_request["id"],
                        "trigger_condition": trigger_condition,
                        "requesting_agent": requesting_agent,
                        "pending_decisions_count": len(state.get("pending_human_decisions", [])) + 1
                    }
                )
                
                return Command(
                    goto="human_approval_handler",  # Special handler node
                    update={
                        "approval_required": True,
                        "approval_context": approval_request,
                        "pending_human_decisions": [
                            *state.get("pending_human_decisions", []),
                            approval_request
                        ],
                        "agent_status_matrix": {
                            **state["agent_status_matrix"],
                            requesting_agent: GeneratorAgentStatus.WAITING
                        }
                    },
                    graph=Command.PARENT
                )
                
            except Exception as e:
                self.logger.log_structured(
                    level="ERROR",
                    message="Approval request failed",
                    extra={
                        "trigger_condition": trigger_condition,
                        "requesting_agent": state.get("active_agent", "unknown"),
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
        
        return approval_checkpoint_tool
    
    def create_human_approval_handler(self):
        """Create human approval handler node"""
        
        def human_approval_handler(state: GeneratorSwarmState) -> Command:
            """Handle human approval workflow"""
            try:
                approval_context = state.get("approval_context", {})
                
                if not approval_context:
                    # No pending approvals, continue normal flow
                    self.logger.log_structured(
                        level="DEBUG",
                        message="No pending approvals, continuing normal flow",
                        extra={"active_agent": state.get("active_agent", "unknown")}
                    )
                    
                    controller = GeneratorStageController()
                    next_agent = controller.determine_next_active_agent(state)
                    return Command(
                        goto=next_agent,
                        update={
                            "approval_required": False,
                            "active_agent": next_agent
                        }
                    )
                
                # Check timeout
                request_time = datetime.datetime.fromisoformat(approval_context["timestamp"])
                timeout_minutes = approval_context["timeout_minutes"]
                elapsed_seconds = (datetime.datetime.now() - request_time).total_seconds()
                
                if elapsed_seconds > timeout_minutes * 60:
                    # Timeout - use default action or escalate
                    self.logger.log_structured(
                        level="WARNING",
                        message="Approval request timed out",
                        extra={
                            "approval_id": approval_context.get("id", "unknown"),
                            "elapsed_minutes": elapsed_seconds / 60,
                            "timeout_minutes": timeout_minutes,
                            "trigger_condition": approval_context.get("trigger_condition", "unknown")
                        }
                    )
                    return self.handle_approval_timeout(state, approval_context)
                
                # Wait for human input (this would integrate with external approval system)
                self.logger.log_structured(
                    level="DEBUG",
                    message="Waiting for human approval input",
                    extra={
                        "approval_id": approval_context.get("id", "unknown"),
                        "elapsed_minutes": elapsed_seconds / 60,
                        "timeout_minutes": timeout_minutes,
                        "status": "awaiting_human_input"
                    }
                )
                
                return Command(
                    goto="human_approval_handler",  # Stay in approval loop
                    update={
                        "approval_context": {
                            **approval_context,
                            "status": "awaiting_human_input"
                        }
                    }
                )
                
            except Exception as e:
                self.logger.log_structured(
                    level="ERROR",
                    message="Human approval handler failed",
                    extra={
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "approval_context": approval_context
                    }
                )
                # Return safe fallback command
                return Command(
                    goto="resource_configuration_agent",
                    update={"active_agent": "resource_configuration_agent"},
                    graph=Command.PARENT
                )
        
        return human_approval_handler
    
    def create_approval_response_tool(self):
        """Tool for processing human approval responses"""
        
        @tool("process_human_approval")
        def approval_response_tool(
            approval_id: Annotated[str, "ID of approval request"],
            decision: Annotated[str, "approved/rejected/modified"],
            feedback: Annotated[str, "Human feedback or modifications"],
            state: Annotated[GeneratorSwarmState, InjectedState]
        ) -> Command:
            """Process human approval response and update state accordingly."""
            try:
                self.logger.log_structured(
                    level="INFO",
                    message="Processing human approval response",
                    extra={
                        "approval_id": approval_id,
                        "decision": decision,
                        "feedback_length": len(feedback) if feedback else 0,
                        "pending_decisions_count": len(state.get("pending_human_decisions", []))
                    }
                )
                
                # Find and update approval request
                updated_decisions = []
                processed_request = None
                
                for decision_item in state.get("pending_human_decisions", []):
                    if decision_item["id"] == approval_id:
                        processed_request = {
                            **decision_item,
                            "status": decision,
                            "human_feedback": feedback,
                            "processed_timestamp": datetime.datetime.now().isoformat()
                        }
                    else:
                        updated_decisions.append(decision_item)
                
                if not processed_request:
                    # Approval ID not found - error condition
                    self.logger.log_structured(
                        level="ERROR",
                        message="Approval ID not found",
                        extra={
                            "approval_id": approval_id,
                            "available_ids": [item["id"] for item in state.get("pending_human_decisions", [])]
                        }
                    )
                    return Command(
                        goto="error_handler",
                        update={"error_context": f"Approval ID {approval_id} not found"}
                    )
            
                # Resume agent execution based on decision
                if decision == "approved":
                    # Resume with original agent
                    requesting_agent = processed_request["requested_by_agent"]
                    
                    self.logger.log_structured(
                        level="INFO",
                        message="Approval granted, resuming agent execution",
                        extra={
                            "approval_id": approval_id,
                            "requesting_agent": requesting_agent,
                            "trigger_condition": processed_request.get("trigger_condition", "unknown")
                        }
                    )
                    
                    return Command(
                        goto=requesting_agent,
                        update={
                            "approval_required": False,
                            "approval_context": {},
                            "pending_human_decisions": updated_decisions,
                            "agent_status_matrix": {
                                **state["agent_status_matrix"],
                                requesting_agent: GeneratorAgentStatus.ACTIVE
                            },
                            "human_approval_log": [
                                *state.get("human_approval_log", []),
                                processed_request
                            ]
                        }
                    )
                elif decision == "rejected":
                    # Handle rejection - potentially reset or escalate
                    self.logger.log_structured(
                        level="WARNING",
                        message="Approval rejected, handling rejection",
                        extra={
                            "approval_id": approval_id,
                            "requesting_agent": processed_request.get("requested_by_agent", "unknown"),
                            "trigger_condition": processed_request.get("trigger_condition", "unknown")
                        }
                    )
                    return self.handle_approval_rejection(state, processed_request)
                else:  # modified
                    # Apply modifications and continue
                    self.logger.log_structured(
                        level="INFO",
                        message="Approval modified, applying changes",
                        extra={
                            "approval_id": approval_id,
                            "requesting_agent": processed_request.get("requested_by_agent", "unknown"),
                            "trigger_condition": processed_request.get("trigger_condition", "unknown")
                        }
                    )
                    return self.handle_approval_modification(state, processed_request, feedback)
                    
            except Exception as e:
                self.logger.log_structured(
                    level="ERROR",
                    message="Approval response processing failed",
                    extra={
                        "approval_id": approval_id,
                        "decision": decision,
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
        
        return approval_response_tool

    def should_request_approval(
        self, 
        context: Dict[str, Any], 
        trigger_type: str
    ) -> bool:
        """Determine if human approval is required"""
        try:
            approval_required = False
            reason = ""
            
            if trigger_type == "high_cost_resources":
                estimated_cost = context.get("estimated_monthly_cost", 0)
                approval_required = estimated_cost > self.approval_thresholds["high_cost_resources"]
                reason = f"Cost ${estimated_cost} exceeds threshold ${self.approval_thresholds['high_cost_resources']}"
            elif trigger_type == "security_critical":
                security_impact = context.get("security_impact", "low")
                approval_required = security_impact in ["high", "critical"]
                reason = f"Security impact: {security_impact}"
            elif trigger_type == "cross_region":
                regions = context.get("regions", [])
                approval_required = len(set(regions)) > 1
                reason = f"Cross-region deployment: {regions}"
            elif trigger_type == "experimental":
                uses_experimental = context.get("uses_experimental_features", False)
                approval_required = uses_experimental
                reason = "Uses experimental features"
            
            self.logger.log_structured(
                level="DEBUG",
                message="Approval requirement check",
                extra={
                    "trigger_type": trigger_type,
                    "approval_required": approval_required,
                    "reason": reason,
                    "context_keys": list(context.keys())
                }
            )
            
            return approval_required
            
        except Exception as e:
            self.logger.log_structured(
                level="ERROR",
                message="Approval requirement check failed",
                extra={
                    "trigger_type": trigger_type,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            return False
    
    def handle_approval_timeout(self, state: GeneratorSwarmState, approval_context: Dict[str, Any]) -> Command:
        """Handle approval timeout scenarios"""
        try:
            self.logger.log_structured(
                level="WARNING",
                message="Handling approval timeout",
                extra={
                    "approval_id": approval_context.get("id", "unknown"),
                    "trigger_condition": approval_context.get("trigger_condition", "unknown"),
                    "requesting_agent": approval_context.get("requested_by_agent", "unknown")
                }
            )
            
            # Default action: escalate or use safe defaults
            requesting_agent = approval_context.get("requested_by_agent", "resource_configuration_agent")
            
            return Command(
                goto=requesting_agent,
                update={
                    "approval_required": False,
                    "approval_context": {},
                    "agent_status_matrix": {
                        **state["agent_status_matrix"],
                        requesting_agent: GeneratorAgentStatus.ACTIVE
                    },
                    "human_approval_log": [
                        *state.get("human_approval_log", []),
                        {
                            **approval_context,
                            "status": "timeout",
                            "processed_timestamp": datetime.datetime.now().isoformat()
                        }
                    ]
                },
                graph=Command.PARENT
            )
            
        except Exception as e:
            self.logger.log_structured(
                level="ERROR",
                message="Approval timeout handling failed",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            return Command(
                goto="resource_configuration_agent",
                update={"active_agent": "resource_configuration_agent"},
                graph=Command.PARENT
            )
    
    def handle_approval_rejection(self, state: GeneratorSwarmState, processed_request: Dict[str, Any]) -> Command:
        """Handle approval rejection scenarios"""
        try:
            self.logger.log_structured(
                level="WARNING",
                message="Handling approval rejection",
                extra={
                    "approval_id": processed_request.get("id", "unknown"),
                    "trigger_condition": processed_request.get("trigger_condition", "unknown"),
                    "requesting_agent": processed_request.get("requested_by_agent", "unknown")
                }
            )
            
            # Default action: reset agent or escalate
            requesting_agent = processed_request.get("requested_by_agent", "resource_configuration_agent")
            
            return Command(
                goto=requesting_agent,
                update={
                    "approval_required": False,
                    "approval_context": {},
                    "agent_status_matrix": {
                        **state["agent_status_matrix"],
                        requesting_agent: GeneratorAgentStatus.INACTIVE  # Reset agent
                    },
                    "human_approval_log": [
                        *state.get("human_approval_log", []),
                        processed_request
                    ]
                },
                graph=Command.PARENT
            )
            
        except Exception as e:
            self.logger.log_structured(
                level="ERROR",
                message="Approval rejection handling failed",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            return Command(
                goto="resource_configuration_agent",
                update={"active_agent": "resource_configuration_agent"},
                graph=Command.PARENT
            )
    
    def handle_approval_modification(self, state: GeneratorSwarmState, processed_request: Dict[str, Any], feedback: str) -> Command:
        """Handle approval modification scenarios"""
        try:
            self.logger.log_structured(
                level="INFO",
                message="Handling approval modification",
                extra={
                    "approval_id": processed_request.get("id", "unknown"),
                    "trigger_condition": processed_request.get("trigger_condition", "unknown"),
                    "requesting_agent": processed_request.get("requested_by_agent", "unknown"),
                    "feedback_length": len(feedback) if feedback else 0
                }
            )
            
            # Apply modifications and continue
            requesting_agent = processed_request.get("requested_by_agent", "resource_configuration_agent")
            
            return Command(
                goto=requesting_agent,
                update={
                    "approval_required": False,
                    "approval_context": {},
                    "agent_status_matrix": {
                        **state["agent_status_matrix"],
                        requesting_agent: GeneratorAgentStatus.ACTIVE
                    },
                    "human_approval_log": [
                        *state.get("human_approval_log", []),
                        processed_request
                    ],
                    "modification_context": {
                        "approval_id": processed_request.get("id"),
                        "modifications": feedback,
                        "applied_timestamp": datetime.datetime.now().isoformat()
                    }
                },
                graph=Command.PARENT
            )
            
        except Exception as e:
            self.logger.log_structured(
                level="ERROR",
                message="Approval modification handling failed",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            return Command(
                goto="resource_configuration_agent",
                update={"active_agent": "resource_configuration_agent"},
                graph=Command.PARENT
            )
