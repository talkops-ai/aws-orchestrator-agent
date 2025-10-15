"""
Utility functions for planner agents to avoid circular imports.
"""

from datetime import datetime
from typing import Dict, Any


def create_agent_completion_data(agent_name: str, task_type: str, data_type: str, status: str = "completed") -> dict:
    """
    Create standardized agent completion data for unified tracking.
    
    Args:
        agent_name: Name of the agent (e.g., 'requirements_analyzer', 'tf_security_n_best_practices_evaluator')
        task_type: Type of task completed (e.g., 'terraform_attribute_mapping', 'security_analysis')
        data_type: Type of data produced (e.g., 'terraform_attribute_mapping', 'security_report')
        status: Completion status (default: 'completed')
        
    Returns:
        dict: Standardized agent completion data
    """
    return {
        "agent_name": agent_name,
        "task_type": task_type,
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "data_type": data_type
    }
