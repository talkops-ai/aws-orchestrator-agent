"""
Global state module for Generator Swarm
"""
from typing import Dict, Any, Optional
from pydantic import BaseModel

# Global state storage
_current_state: Optional[Dict[str, Any]] = None

def set_current_state(state: Dict[str, Any]) -> None:
    """Set the current global state"""
    global _current_state
    _current_state = state

def get_current_state() -> Optional[Dict[str, Any]]:
    """Get the current global state"""
    return _current_state

def update_current_state(updates: Dict[str, Any]) -> None:
    """Update the current global state by merging with provided updates"""
    global _current_state
    if _current_state is None:
        _current_state = updates.copy()
    else:
        _current_state.update(updates)

def merge_current_state(updates: Dict[str, Any]) -> None:
    """Deep merge updates into the current global state"""
    global _current_state
    if _current_state is None:
        _current_state = updates.copy()
    else:
        _current_state = deep_merge(_current_state, updates)

def update_agent_workspace(agent_name: str, workspace_updates: Dict[str, Any]) -> None:
    """Update a specific agent's workspace in the global state"""
    global _current_state
    if _current_state is None:
        _current_state = {}
    
    if "agent_workspaces" not in _current_state:
        _current_state["agent_workspaces"] = {}
    
    if agent_name not in _current_state["agent_workspaces"]:
        _current_state["agent_workspaces"][agent_name] = {}
    
    # Convert Pydantic objects to dictionaries automatically
    serialized_updates = _serialize_pydantic_objects(workspace_updates)
    _current_state["agent_workspaces"][agent_name].update(serialized_updates)

def _serialize_pydantic_objects(data: Any) -> Any:
    """Recursively convert Pydantic objects to dictionaries"""
    if isinstance(data, BaseModel):
        return data.model_dump()
    elif isinstance(data, list):
        return [_serialize_pydantic_objects(item) for item in data]
    elif isinstance(data, dict):
        return {key: _serialize_pydantic_objects(value) for key, value in data.items()}
    else:
        return data

def deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries"""
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def clear_current_state() -> None:
    """Clear the current global state"""
    global _current_state
    _current_state = None
