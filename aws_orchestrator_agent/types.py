"""
Base types for AWS Orchestrator Agent.

This module defines the core interfaces and data structures used throughout
the AWS Orchestrator Agent system.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, Optional, Union, List

from pydantic import BaseModel, Field, ConfigDict


class AgentResponse(BaseModel):
    """
    Response from an agent during execution.
    
    This represents a single response item from the agent's stream,
    containing the content, metadata, and control flags.
    """
    
    model_config = ConfigDict(extra="allow")
    
    content: Any = Field(..., description="The response content (text or data)")
    response_type: str = Field(default="text", description="Type of response: 'text' or 'data'")
    is_task_complete: bool = Field(default=False, description="Whether this response indicates task completion")
    require_user_input: bool = Field(default=False, description="Whether this response requires user input to continue")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the response")
    root: Optional[Any] = Field(default=None, description="Root object for A2A protocol integration")


class BaseAgent(ABC):
    """
    Base interface for all AWS Orchestrator Agent implementations.
    
    This abstract base class defines the contract that all agent implementations
    must follow to work with the A2A protocol integration.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of the agent.
        
        Returns:
            The agent's name
        """
        pass
    
    @abstractmethod
    async def stream(
        self, 
        query: str, 
        context_id: str, 
        task_id: str
    ) -> AsyncGenerator[AgentResponse, None]:
        """
        Stream responses for a given query.
        
        This method should implement the core agent logic and yield
        AgentResponse objects as the agent processes the query.
        
        Args:
            query: The user query to process
            context_id: The A2A context ID
            task_id: The A2A task ID
            
        Yields:
            AgentResponse objects representing the agent's progress
        """
        pass
    
    async def initialize(self) -> None:
        """
        Initialize the agent.
        
        This method can be overridden to perform any initialization
        required by the agent implementation.
        """
        pass
    
    async def cleanup(self) -> None:
        """
        Clean up resources used by the agent.
        
        This method can be overridden to perform any cleanup
        required by the agent implementation.
        """
        pass


class AgentConfig(BaseModel):
    """
    Configuration for an agent.
    
    This contains all the configuration parameters needed to
    initialize and run an agent.
    """
    
    model_config = ConfigDict(extra="allow")
    
    name: str = Field(..., description="The name of the agent")
    agent_type: str = Field(..., description="The type of agent (e.g., 'generation', 'cost_analysis', etc.)")
    enabled: bool = Field(default=True, description="Whether the agent is enabled")
    max_concurrent_tasks: int = Field(default=10, description="Maximum number of concurrent tasks this agent can handle")
    timeout_seconds: int = Field(default=300, description="Timeout for agent execution in seconds")
    config: Dict[str, Any] = Field(default_factory=dict, description="Additional configuration parameters specific to the agent type")


class TaskResult(BaseModel):
    """
    Result of a task execution.
    
    This represents the final result of a task, including
    success status, output, and any errors.
    """
    
    model_config = ConfigDict(extra="allow")
    
    task_id: str = Field(..., description="The ID of the task")
    success: bool = Field(..., description="Whether the task completed successfully")
    output: Optional[str] = Field(default=None, description="The output of the task")
    error: Optional[str] = Field(default=None, description="Error message if the task failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the task result")
    artifacts: List[Any] = Field(default_factory=list, description="List of artifacts produced by the task")
