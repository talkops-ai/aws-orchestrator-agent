"""
Approval Middleware for Human-in-the-Loop Integration
Generation Agent Swarm - Planning Stage

This module provides centralized approval checking logic for all sub-agents
in the Generation Agent Swarm, implementing the recommended Middleware Pattern
from the HITL Integration Guide.
"""

from typing import Dict, Any, Optional, List
from aws_orchestrator_agent.utils.logger import AgentLogger
from .generation_hitl import GeneratorStageHumanLoop


class ApprovalMiddleware:
    """
    Centralized approval checking middleware for the Generation Agent Swarm.
    
    This class provides a unified interface for all sub-agents to request
    human approval for high-risk, high-cost, or security-critical operations.
    It implements the recommended Middleware Pattern from the HITL Integration Guide.
    """
    
    def __init__(self, human_loop: GeneratorStageHumanLoop):
        """
        Initialize the Approval Middleware.
        
        Args:
            human_loop: The GeneratorStageHumanLoop instance for approval management
        """
        self.human_loop = human_loop
        self.logger = AgentLogger("ApprovalMiddleware")
        
        # Cache approval tools for performance
        self._approval_tools_cache = {}
        
        self.logger.log_structured(
            level="INFO",
            message="ApprovalMiddleware initialized",
            extra={
                "human_loop_available": human_loop is not None
            }
        )
    
    async def check_approval(
        self, 
        agent_name: str, 
        action: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check if approval is required for an action and request it if needed.
        
        Args:
            agent_name: Name of the agent requesting approval
            action: The action being performed (e.g., 'create_resource', 'create_variable')
            context: Context information about the action
            
        Returns:
            Dict containing approval result with keys:
            - status: 'approved', 'rejected', 'modified', or 'timeout'
            - reason: Explanation of the decision
            - modifications: Any modifications requested by human (if status is 'modified')
        """
        
        try:
            # Determine trigger type based on agent and action
            trigger_type = self.determine_trigger_type(agent_name, action, context)
            
            if not trigger_type:
                return {"status": "approved", "reason": "No approval required"}
            
            # Check if approval is required based on thresholds
            if not self.human_loop.should_request_approval(context, trigger_type):
                return {"status": "approved", "reason": "Below approval threshold"}
            
            # Get or create approval tool
            approval_tool = self._get_approval_tool(trigger_type)
            
            self.logger.log_structured(
                level="INFO",
                message="Requesting approval for action",
                extra={
                    "agent_name": agent_name,
                    "action": action,
                    "trigger_type": trigger_type,
                    "context_keys": list(context.keys()),
                    "urgency_level": self.calculate_urgency(trigger_type, context)
                }
            )
            
            # Request approval
            approval_result = await approval_tool(
                approval_context=context,
                urgency_level=self.calculate_urgency(trigger_type, context)
            )
            
            self.logger.log_structured(
                level="INFO",
                message="Approval request completed",
                extra={
                    "agent_name": agent_name,
                    "action": action,
                    "trigger_type": trigger_type,
                    "approval_status": approval_result.get("status"),
                    "approval_reason": approval_result.get("reason")
                }
            )
            
            return approval_result
            
        except Exception as e:
            self.logger.log_structured(
                level="ERROR",
                message="Approval check failed",
                extra={
                    "agent_name": agent_name,
                    "action": action,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            # Default to approved on error to avoid blocking operations
            return {"status": "approved", "reason": "Error in approval check, defaulting to approved"}
    
    def determine_trigger_type(self, agent_name: str, action: str, context: Dict[str, Any]) -> Optional[str]:
        """
        Determine the trigger type for approval based on agent, action, and context.
        
        Args:
            agent_name: Name of the agent
            action: The action being performed
            context: Context information
            
        Returns:
            Trigger type string or None if no approval needed
        """
        
        # Resource Configuration Agent
        if agent_name == "resource_configuration_agent":
            if action == "create_resource":
                if context.get("estimated_monthly_cost", 0) > 1000:
                    return "high_cost_resources"
                if self._is_security_critical_resource(context):
                    return "security_critical"
                if self._spans_multiple_regions(context):
                    return "cross_region"
                if self._uses_experimental_features(context):
                    return "experimental"
        
        # Variable Definition Agent
        elif agent_name == "variable_definition_agent":
            if action == "create_variable":
                if context.get("sensitive", False):
                    return "security_critical"
                if self._has_high_cost_impact(context):
                    return "high_cost_resources"
                if self._uses_experimental_types(context):
                    return "experimental"
        
        # Data Source Agent
        elif agent_name == "data_source_agent":
            if action == "create_data_source":
                if self._spans_multiple_regions(context):
                    return "cross_region"
                if context.get("estimated_query_cost", 0) > 1000:
                    return "high_cost_resources"
                if self._uses_experimental_apis(context):
                    return "experimental"
        
        # Local Values Agent
        elif agent_name == "local_values_agent":
            if action == "create_local_value":
                if self._is_complex_computation(context):
                    return "high_cost_resources"
                if self._involves_security_calculations(context):
                    return "security_critical"
                if self._uses_experimental_functions(context):
                    return "experimental"
        
        return None
    
    def calculate_urgency(self, trigger_type: str, context: Dict[str, Any]) -> int:
        """
        Calculate urgency level for approval request.
        
        Args:
            trigger_type: Type of trigger requiring approval
            context: Context information
            
        Returns:
            Urgency level (1-5, where 5 is highest)
        """
        base_urgency = {
            "high_cost_resources": 3,
            "security_critical": 4,
            "cross_region": 2,
            "experimental": 3
        }
        
        urgency = base_urgency.get(trigger_type, 3)
        
        # Adjust urgency based on context
        if trigger_type == "high_cost_resources":
            cost = context.get("estimated_monthly_cost", 0)
            if cost > 5000:
                urgency = 4
            elif cost > 10000:
                urgency = 5
        
        elif trigger_type == "security_critical":
            # Higher urgency for more sensitive operations
            if context.get("sensitive", False):
                urgency = 5
        
        return urgency
    
    def _get_approval_tool(self, trigger_type: str):
        """
        Get or create approval tool for the given trigger type.
        
        Args:
            trigger_type: Type of trigger
            
        Returns:
            Approval tool function
        """
        if trigger_type not in self._approval_tools_cache:
            self._approval_tools_cache[trigger_type] = self.human_loop.create_approval_checkpoint_tool(trigger_type)
        
        return self._approval_tools_cache[trigger_type]
    
    # Helper methods for trigger type determination
    
    def _is_security_critical_resource(self, context: Dict[str, Any]) -> bool:
        """Check if resource is security-critical"""
        security_types = [
            "aws_iam_role", "aws_iam_policy", "aws_security_group",
            "aws_kms_key", "aws_secretsmanager_secret", "aws_iam_instance_profile",
            "aws_iam_policy_attachment", "aws_iam_role_policy_attachment"
        ]
        return context.get("type") in security_types
    
    def _spans_multiple_regions(self, context: Dict[str, Any]) -> bool:
        """Check if operation spans multiple regions"""
        regions = context.get("regions", [])
        if isinstance(regions, list):
            return len(set(regions)) > 1
        return False
    
    def _uses_experimental_features(self, context: Dict[str, Any]) -> bool:
        """Check if operation uses experimental features"""
        return context.get("uses_experimental_features", False)
    
    def _has_high_cost_impact(self, context: Dict[str, Any]) -> bool:
        """Check if variable has high cost impact"""
        return context.get("cost_impact", "low") in ["high", "critical"]
    
    def _uses_experimental_types(self, context: Dict[str, Any]) -> bool:
        """Check if operation uses experimental types"""
        return context.get("uses_experimental_types", False)
    
    def _uses_experimental_apis(self, context: Dict[str, Any]) -> bool:
        """Check if operation uses experimental APIs"""
        return context.get("uses_experimental_apis", False)
    
    def _is_complex_computation(self, context: Dict[str, Any]) -> bool:
        """Check if local value involves complex computation"""
        complexity = context.get("complexity", "low")
        return complexity in ["high", "critical"]
    
    def _involves_security_calculations(self, context: Dict[str, Any]) -> bool:
        """Check if local value involves security calculations"""
        return context.get("security_related", False)
    
    def _uses_experimental_functions(self, context: Dict[str, Any]) -> bool:
        """Check if operation uses experimental functions"""
        return context.get("uses_experimental_functions", False)
    
    def get_approval_summary(self) -> Dict[str, Any]:
        """
        Get summary of approval middleware configuration.
        
        Returns:
            Dict containing middleware configuration summary
        """
        return {
            "middleware_type": "ApprovalMiddleware",
            "human_loop_available": self.human_loop is not None,
            "cached_tools": list(self._approval_tools_cache.keys()),
            "supported_trigger_types": [
                "high_cost_resources",
                "security_critical", 
                "cross_region",
                "experimental"
            ],
            "supported_agents": [
                "resource_configuration_agent",
                "variable_definition_agent",
                "data_source_agent",
                "local_values_agent"
            ]
        }
