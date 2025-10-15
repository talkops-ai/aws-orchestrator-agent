"""
Base agent abstract class for LangGraph subgraph agents.

This module provides the base abstract class that all subgraph agents must inherit from
when working with the custom supervisor agent using the Send() pattern.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from langgraph.graph import StateGraph


class BaseSubgraphAgent(ABC):
    """
    Base abstract class for subgraph agents using LangGraph Send() pattern.
    
    All specialized agents (Generation, Editor, Validation, etc.) must inherit
    from this class and implement the required abstract methods.
    
    This class provides:
    - Standard interface for supervisor integration
    - State transformation between supervisor and agent
    - Human-in-the-loop interruption support
    - LangGraph subgraph compatibility
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Agent name for Send() routing and identification.
        
        Returns:
            str: Unique name for this agent (e.g., "generation_agent", "editor_agent")
        """
        pass
    
    @property
    @abstractmethod
    def state_model(self) -> type[BaseModel]:
        """
        Pydantic model for agent's state schema.
        
        Returns:
            type[BaseModel]: The Pydantic model class for this agent's state
        """
        pass
    
    @property
    def memory(self):
        """
        Memory/checkpointer instance for this agent.
        
        Returns:
            MemorySaver: The checkpointer instance used by this agent
        """
        return getattr(self, '_memory', None)
    
    @memory.setter
    def memory(self, value):
        """
        Set the memory/checkpointer for this agent.
        
        Args:
            value: MemorySaver instance to use for this agent
        """
        self._memory = value
    
    @abstractmethod
    def build_graph(self) -> StateGraph:
        """
        Build the LangGraph StateGraph for this agent.
        
        This method should:
        1. Create a StateGraph with the agent's state model
        2. Add nodes for the agent's workflow
        3. Define edges between nodes
        4. Set entry point and exit conditions
        
        Returns:
            StateGraph: The compiled graph for this agent
        """
        pass
    
    def input_transform(self, send_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform Send() payload from supervisor to agent state.
        
        The supervisor uses Send() to pass data to subgraph agents. This method
        extracts the task description and transforms it into the agent's state format.
        
        Args:
            send_payload: Data sent from supervisor via Send() primitive
                - messages: List of messages, typically [{"role": "user", "content": task_description}]
                - Any other data included in the Send() call
        
        Returns:
            Dict[str, Any]: Transformed state ready for agent processing
        """
        # Extract task description from Send payload
        task_description = ""
        if "messages" in send_payload and send_payload["messages"]:
            task_description = send_payload["messages"][0].get("content", "")
        
        # Transform to agent's state model format
        return {
            "user_request": task_description,
            "messages": send_payload.get("messages", []),
            # Add any other fields your agent state needs
            "supervisor_context": send_payload.get("context", {}),
        }
    
    def output_transform(self, agent_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform agent state back to supervisor state.
        
        This method prepares the agent's result for merging back into the
        supervisor's state. Only include data that should propagate to supervisor.
        
        Args:
            agent_state: The final state from agent execution
        
        Returns:
            Dict[str, Any]: Data to merge into supervisor state
        """
        return {
            "messages": agent_state.get("messages", []),
            "agent_result": agent_state.get("result", {}),
            "agent_status": agent_state.get("status", "completed"),
            "agent_metadata": {
                "agent_name": self.name,
                "execution_time": agent_state.get("execution_time"),
                "warnings": agent_state.get("warnings", []),
                "errors": agent_state.get("errors", []),
            }
        }
    
    def can_interrupt(self) -> bool:
        """
        Whether this agent can interrupt for human input.
        
        Override this method to return True if your agent needs
        human-in-the-loop capabilities.
        
        Returns:
            bool: True if agent can interrupt, False otherwise
        """
        return False
    
    def handle_interrupt(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle human-in-the-loop interruption.
        
        When an agent needs human input, it should:
        1. Set interrupt flag in state
        2. Return state with interrupt context
        3. Supervisor will pause and wait for human input
        
        Args:
            state: Current agent state
        
        Returns:
            Dict[str, Any]: State with interrupt information
        """
        # Set interrupt flag
        state["interrupt_required"] = True
        state["interrupt_context"] = {
            "agent": self.name,
            "reason": "human_input_required",
            "data": state.get("interrupt_data", {}),
            "timestamp": state.get("timestamp"),
        }
        return state
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """
        Get tools available to this agent.
        
        Override this method to expose agent-specific tools
        that can be called by the supervisor or other agents.
        
        Returns:
            List[Dict[str, Any]]: List of tool specifications
        """
        return []
    
    def validate_state(self, state: Dict[str, Any]) -> bool:
        """
        Validate agent state before processing.
        
        Override this method to add custom validation logic
        for your agent's state.
        
        Args:
            state: Agent state to validate
        
        Returns:
            bool: True if state is valid, False otherwise
        """
        try:
            # Try to create state model instance to validate
            self.state_model(**state)
            return True
        except Exception:
            return False
    
    def preprocess_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess state before agent execution.
        
        Override this method to add preprocessing logic
        like setting default values, validating inputs, etc.
        
        Args:
            state: Raw agent state
        
        Returns:
            Dict[str, Any]: Preprocessed state
        """
        return state
    
    def postprocess_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Postprocess result after agent execution.
        
        Override this method to add postprocessing logic
        like formatting outputs, adding metadata, etc.
        
        Args:
            result: Raw agent result
        
        Returns:
            Dict[str, Any]: Postprocessed result
        """
        return result


# Type alias for convenience
AgentType = BaseSubgraphAgent
