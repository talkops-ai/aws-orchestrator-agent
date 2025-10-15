"""
Planner Sub-Supervisor Agent using langgraph-supervisor.

This module implements the Planner Sub-Supervisor Agent, which manages the planning
sub-agents (Requirements Analyzer, Dependency Mapper, Execution Planner) using
langgraph-supervisor with custom handoff tools.

The sub-supervisor coordinates the planning workflow and routes between sub-agents
based on the current state of the shared PlannerSupervisorState.
"""

import json
import traceback
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import re
from functools import wraps
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph_supervisor import create_supervisor
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from aws_orchestrator_agent.core.llm.llm_provider import LLMProvider
from aws_orchestrator_agent.config.config import Config
from aws_orchestrator_agent.utils.logger import AgentLogger, log_sync
from aws_orchestrator_agent.core.agents.base_agent import BaseSubgraphAgent

from .planner_supervisor_state import (
    PlannerSupervisorState,
    create_initial_planner_state,
    create_planning_results
)
from .planner_handoff_tools import create_planner_handoff_tools
from .sub_agents import (
    create_requirements_analyzer_react_agent,
    #create_security_n_best_practices_react_agent,
    create_execution_planner_react_agent
)

# Create agent logger for planner sub-supervisor
planner_supervisor_logger = AgentLogger("PLANNER_SUPERVISOR")


class PlannerSubSupervisorAgent(BaseSubgraphAgent):
    """
    Planner Sub-Supervisor Agent that manages planning sub-agents using langgraph-supervisor.
    
    This agent coordinates the planning workflow by routing between specialized
    sub-agents using custom handoff tools and the langgraph-supervisor library.
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        custom_config: Optional[Dict[str, Any]] = None,
        name: str = "planner_sub_supervisor_agent",
        memory: Optional[MemorySaver] = None
    ):
        """
        Initialize the Planner Sub-Supervisor Agent.
        
        Args:
            config: Configuration instance (defaults to new Config())
            custom_config: Optional custom configuration to override defaults
            name: Agent name for identification
            memory: Shared memory/checkpointer instance
        """
        planner_supervisor_logger.log_structured(
            level="INFO",
            message="=== PLANNER SUB SUPERVISOR INITIALIZATION START ===",
            extra={
                "name": name,
                "has_config": config is not None,
                "has_custom_config": custom_config is not None,
                "has_memory": memory is not None
            }
        )
        
        # Use centralized config system
        self.config_instance = config or Config(custom_config or {})
        
        # Set agent name for identification
        self._name = name
        
        self._planner_supervisor_state = PlannerSupervisorState()
        
        # Set shared memory
        self.memory = memory or MemorySaver()
        
        planner_supervisor_logger.log_structured(
            level="DEBUG",
            message="Basic initialization complete",
            extra={
                "config_type": type(self.config_instance).__name__,
                "memory_type": type(self.memory).__name__
            }
        )
        
        # Get LLM configuration from centralized config
        llm_config = self.config_instance.get_llm_config()
        
        planner_supervisor_logger.log_structured(
            level="DEBUG",
            message="LLM config retrieved",
            extra={
                "llm_provider": llm_config.get('provider'),
                "llm_model": llm_config.get('model'),
                "llm_temperature": llm_config.get('temperature'),
                "llm_max_tokens": llm_config.get('max_tokens')
            }
        )
        
        # Initialize the LLM model using the centralized provider
        try:
            self.model = LLMProvider.create_llm(
                provider=llm_config['provider'],
                model=llm_config['model'],
                temperature=llm_config['temperature'],
                max_tokens=llm_config['max_tokens']
            )
            planner_supervisor_logger.log_structured(
                level="INFO",
                message="LLM model initialized successfully",
                extra={
                    "llm_provider": llm_config['provider'], 
                    "llm_model": llm_config['model'],
                    "model_type": type(self.model).__name__
                }
            )
        except Exception as e:
            planner_supervisor_logger.log_structured(
                level="ERROR",
                message="LLM model initialization failed",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "llm_provider": llm_config.get('provider'),
                    "llm_model": llm_config.get('model')
                }
            )
            raise
        
        # Initialize sub-agents
        planner_supervisor_logger.log_structured(
            level="DEBUG",
            message="Starting sub-agents initialization",
            extra={"config_type": type(self.config_instance).__name__}
        )
        
        self._initialize_sub_agents()
        
        # Create handoff tools
        planner_supervisor_logger.log_structured(
            level="DEBUG",
            message="Creating handoff tools",
            extra={}
        )
        
        self.handoff_tools = create_planner_handoff_tools()
        
        planner_supervisor_logger.log_structured(
            level="DEBUG",
            message="Handoff tools created",
            extra={
                "handoff_tools_count": len(self.handoff_tools),
                "handoff_tools_names": list(self.handoff_tools.keys())
            }
        )
        
        # Define supervisor prompt
        planner_supervisor_logger.log_structured(
            level="DEBUG",
            message="Defining supervisor prompt",
            extra={}
        )
        
        self._define_supervisor_prompt()
        
        planner_supervisor_logger.log_structured(
            level="INFO",
            message="=== PLANNER SUB SUPERVISOR INITIALIZATION COMPLETE ===",
            extra={
                "llm_provider": llm_config['provider'],
                "llm_model": llm_config['model'],
                "name": name,
                "sub_agents_count": 3,
                "handoff_tools_count": len(self.handoff_tools),
                "supervisor_prompt_defined": hasattr(self, 'supervisor_prompt')
            }
        )
    
    @property
    def name(self) -> str:
        """Agent name for Send() routing and identification."""
        return self._name
    
    @property
    def state_model(self) -> type[BaseModel]:
        """Get the state model for this agent."""
        return PlannerSupervisorState
    
    def _initialize_sub_agents(self):
        """Initialize the planning sub-agents."""
        try:
            planner_supervisor_logger.log_structured(
                level="DEBUG",
                message="=== SUB-AGENTS INITIALIZATION START ===",
                extra={
                    "config_type": type(self.config_instance).__name__
                }
            )
            
            # Create React agents using factory functions
            planner_supervisor_logger.log_structured(
                level="DEBUG",
                message="Creating requirements analyzer",
                extra={}
            )
            
            self.requirements_analyzer = create_requirements_analyzer_react_agent(state=self._planner_supervisor_state, config=self.config_instance)
            
            planner_supervisor_logger.log_structured(
                level="DEBUG",
                message="Requirements analyzer created",
                extra={
                    "requirements_analyzer_type": type(self.requirements_analyzer).__name__
                }
            )
            
            # planner_supervisor_logger.log_structured(
            #     level="DEBUG",
            #     message="Creating dependency mapper",
            #     extra={}
            # )
            
            # self.security_n_best_practices_evaluator = create_security_n_best_practices_react_agent(state=self._planner_supervisor_state, config=self.config_instance)
            
            # planner_supervisor_logger.log_structured(
            #     level="DEBUG",
            #     message="tf_security_n_best_practices_evaluator created",
            #     extra={
            #         "security_n_best_practices_evaluator_type": type(self.security_n_best_practices_evaluator).__name__
            #     }
            # )
            
            planner_supervisor_logger.log_structured(
                level="DEBUG",
                message="Creating execution planner",
                extra={}
            )
            
            self.execution_planner = create_execution_planner_react_agent(state=self._planner_supervisor_state, config=self.config_instance)
            
            planner_supervisor_logger.log_structured(
                level="DEBUG",
                message="Execution planner created",
                extra={
                    "execution_planner_type": type(self.execution_planner).__name__
                }
            )
            
            planner_supervisor_logger.log_structured(
                level="INFO",
                message="=== SUB-AGENTS INITIALIZATION COMPLETE ===",
                extra={
                    "requirements_analyzer": "requirements_analyzer",
                    "execution_planner": "execution_planner",
                    "all_agents_created": all([
                        hasattr(self, 'requirements_analyzer'),
                        hasattr(self, 'execution_planner')
                    ])
                }
            )
            
        except Exception as e:
            planner_supervisor_logger.log_structured(
                level="ERROR",
                message="=== SUB-AGENTS INITIALIZATION FAILED ===",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc()
                }
            )
            raise
    
    def _define_supervisor_prompt(self):
        """Define the supervisor prompt for langgraph-supervisor."""
        self.supervisor_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a planning supervisor that coordinates between specialized planning agents.

Your role is to:
1. Analyze the user's infrastructure request
2. Route the request to the appropriate planning agent
3. Coordinate the planning workflow
4. Ensure all planning phases are completed

Available agents:
- requirements_analyzer: Analyzes business and technical requirements
- execution_planner: Creates execution plans and assesses risks

Available handoff tools:
- handoff_to_requirements_analyzer: Transfer to requirements analysis
- handoff_to_execution_planner: Transfer to execution planning
- mark_planning_complete: Mark planning as complete and set completion flags (call this first when all phases are done)
- handoff_to_planner_complete: Return to main supervisor (call this after mark_planning_complete)

Planning workflow:
1. Start with requirements analysis (ALWAYS start here)
2. Move to execution planning (may require user input)
3. Complete with execution planning
4. When all phases are done: FIRST call mark_planning_complete, THEN call handoff_to_planner_complete

IMPORTANT INSTRUCTIONS:
- ALWAYS start the planning workflow by handing off to requirements_analyzer
- If the user request is empty or unclear, still start with requirements analysis
- The requirements_analyzer will handle gathering requirements from the user if needed
- Use the handoff tools to transfer control to the appropriate agent
- Always provide clear task descriptions when handing off
- Do not ask the user for input directly - let the specialized agents handle that

STATE HANDLING:
- When you receive a handoff from the main supervisor, the state will contain:
  * user_request: The original user request (e.g., "can you help me in writing aws kms module")
  * task_description: The task description from the main supervisor
  * session_id: Session identifier for tracking
  * task_id: Task identifier for tracking
- ALWAYS use the user_request from the state when handing off to requirements_analyzer
- If user_request is empty, use the task_description as the user request
- Pass the session_id and task_id through to maintain context

ROUTING DECISIONS:
1. Check planning_workflow_state.loop_counter - if > 10, terminate with error
2. Check if planning is complete:
   - If planning_workflow_state.planning_complete == True:
     → mark_planning_complete (FIRST) → handoff_to_planner_complete (SECOND)
3. Check planning_workflow_state.next_phase to determine routing:
   - If next_phase == "requirements_analysis" and not requirements_complete:
     → handoff_to_requirements_analyzer
   - If next_phase == "execution_planning" and requirements_complete:
     → handoff_to_execution_planner
   - If next_phase == None (all phases complete):
     → mark_planning_complete (FIRST) → handoff_to_planner_complete (SECOND)

COMPLETION VALIDATION:
- After each agent handoff, validate that the expected completion flag is set
- Check if planning_workflow_state.planning_complete == True to determine if all phases are done
- If completion flag not set after reasonable time, log error and retry once
- If retry fails, terminate workflow and escalate to human
- **CRITICAL: When planning_complete == True, you MUST call mark_planning_complete FIRST, then handoff_to_planner_complete**
- **DO NOT continue processing after planning is complete - ALWAYS call both tools in sequence**

ERROR HANDLING:
- Loop counter exceeded: Terminate with "Maximum iterations reached"
- Missing completion flag: Log error and retry once
- Invalid state transition: Log error and terminate

When you receive a request:
1. Check the state for user_request and task_description
2. Check planning_workflow_state.next_phase to determine the correct agent to hand off to
3. Only hand off to requirements_analyzer if requirements_complete = False
4. Let the specialized agent determine if more information is needed
5. Continue the workflow based on completion status
6. **MANDATORY: When all phases are complete (planning_complete == True), you MUST call mark_planning_complete FIRST, then handoff_to_planner_complete**

**TOOL CALL REQUIREMENTS:**
- You MUST use the handoff tools to transfer control - do not just return text
- When planning_workflow_state.planning_complete == True, call mark_planning_complete FIRST, then handoff_to_planner_complete
- Do not provide text responses when you should be calling tools
- The handoff tools are the ONLY way to properly complete the planning workflow
- **CRITICAL: Always call mark_planning_complete before handoff_to_planner_complete to prevent infinite loops**"""),
            ("human", "{user_request}")
        ])
    
    def _process_subsequent_request(self, state: PlannerSupervisorState) -> None:
        """
        Process subsequent requests by checking for agent completions and updating state.
        
        Args:
            state: The current planner supervisor state
        """
        # Log the processing
        planner_supervisor_logger.log_structured(
            level="DEBUG",
            message="Processing subsequent request - checking for agent outputs",
            extra={
                "current_phase": getattr(state.planning_workflow_state, 'current_phase', ''),
                "planning_complete": getattr(state.planning_workflow_state, 'planning_complete', False)
            }
        )
        
        # Increment loop counter and check for errors
        state.increment_loop_counter()
        if state.planning_workflow_state.error_occurred:
            planner_supervisor_logger.log_structured(
                level="ERROR",
                message="Loop limit exceeded in pre-model hook",
                extra={
                    "loop_counter": state.planning_workflow_state.loop_counter,
                    "error_message": state.planning_workflow_state.error_message
                }
            )
            return

        if hasattr(self._planner_supervisor_state.requirements_data, 'terraform_attribute_mapping'):
            transform_data = self._planner_supervisor_state.requirements_data.terraform_attribute_mapping
            if isinstance(transform_data, dict) and 'agent_completion' in transform_data:
                completion_info = transform_data['agent_completion']
                if completion_info.get('status') == 'completed':
                    self._handle_requirements_analyzer_completion(state, transform_data, completion_info)
        
        # Debug: Check if execution completion is detected
        has_agent_completion = hasattr(self._planner_supervisor_state.execution_data, 'agent_completion')
        planner_supervisor_logger.log_structured(
            level="DEBUG",
            message="Checking execution planner completion",
            extra={
                "has_agent_completion": has_agent_completion,
                "execution_plan_complete": getattr(self._planner_supervisor_state.execution_data, 'execution_plan_complete', False),
                "workflow_execution_complete": getattr(self._planner_supervisor_state.planning_workflow_state, 'execution_complete', False)
            }
        )
        
        if has_agent_completion:
            completion_info = self._planner_supervisor_state.execution_data.agent_completion
            if completion_info and completion_info.get('status') == 'completed':
                execution_data = self._planner_supervisor_state.execution_data.execution_plan_data
                planner_supervisor_logger.log_structured(
                    level="INFO",
                    message="Execution planner completion detected - calling handler",
                    extra={
                        "completion_status": completion_info.get('status'),
                        "has_execution_data": bool(execution_data)
                    }
                )
                self._handle_execution_planner_completion(state, execution_data, completion_info)
        
        # Check if workflow is complete after processing agent completions
        if self._is_workflow_complete(state):
            # Handle workflow completion and prepare for Supervisor handoff
            self._handle_workflow_completion(state)
            
            # CRITICAL DEBUG: Log the state that will be passed to the LLM
            planner_supervisor_logger.log_structured(
                level="INFO",
                message="State being passed to LLM after workflow completion",
                extra={
                    "planning_complete": getattr(state.planning_workflow_state, 'planning_complete', False),
                    "execution_complete": getattr(state.planning_workflow_state, 'execution_complete', False),
                    "requirements_complete": getattr(state.planning_workflow_state, 'requirements_complete', False),
                    "current_phase": getattr(state.planning_workflow_state, 'current_phase', ''),
                    "next_phase": getattr(state.planning_workflow_state, 'next_phase', None),
                    "status": getattr(state, 'status', ''),
                    "active_agent": getattr(state, 'active_agent', ''),
                    "should_call_handoff_tool": getattr(state.planning_workflow_state, 'planning_complete', False)
                }
            )
            
            # CRITICAL: Add explicit instruction to LLM when planning is complete
            if getattr(state.planning_workflow_state, 'planning_complete', False):
                planner_supervisor_logger.log_structured(
                    level="WARNING",
                    message="PLANNING COMPLETE - LLM MUST CALL mark_planning_complete THEN handoff_to_planner_complete",
                    extra={
                        "planning_complete": True,
                        "instruction": "LLM should call mark_planning_complete FIRST, then handoff_to_planner_complete",
                        "current_phase": getattr(state.planning_workflow_state, 'current_phase', ''),
                        "next_phase": getattr(state.planning_workflow_state, 'next_phase', None)
                    }
                )

    def _is_workflow_complete(self, state: PlannerSupervisorState) -> bool:
        """
        Check if the planning workflow is complete and ready for Supervisor handoff.
        
        Args:
            state: The current planner supervisor state
            
        Returns:
            bool: True if workflow is complete, False otherwise
        """
        # Check if execution planning is complete in the workflow state (this is what the supervisor checks)
        execution_complete = getattr(state.planning_workflow_state, 'execution_complete', False)
        
        # Check if requirements analysis is complete (prerequisite)
        requirements_complete = getattr(state.planning_workflow_state, 'requirements_complete', False)
        
        # Workflow is complete when both phases are done
        is_complete = execution_complete and requirements_complete
        
        planner_supervisor_logger.log_structured(
            level="DEBUG",
            message="Workflow completion check",
            extra={
                "execution_complete": execution_complete,
                "requirements_complete": requirements_complete,
                "workflow_complete": is_complete,
                "current_phase": getattr(state.planning_workflow_state, 'current_phase', ''),
                "planning_complete": getattr(state.planning_workflow_state, 'planning_complete', False)
            }
        )
        
        return is_complete

    def _handle_workflow_completion(self, state: PlannerSupervisorState) -> None:
        """
        Handle workflow completion and prepare for Supervisor handoff.
        
        Args:
            state: The current planner supervisor state
        """
        planner_supervisor_logger.log_structured(
            level="INFO",
            message="Planning workflow complete - preparing Supervisor handoff",
            extra={
                "completion_timestamp": datetime.now().isoformat(),
                "execution_plan_complete": getattr(state.execution_data, 'execution_plan_complete', False),
                "requirements_complete": getattr(state.requirements_data, 'terraform_attribute_mapping_complete', False),
                "current_phase": getattr(state.planning_workflow_state, 'current_phase', '')
            }
        )
        
        # Mark workflow as complete
        state.planning_workflow_state.planning_complete = True
        # Note: next_phase is a computed property, so we don't set it directly
        
        # Update status to indicate completion
        state.status = "completed"
        
        # Log final state
        planner_supervisor_logger.log_structured(
            level="INFO",
            message="Planning workflow marked as complete - ready for Supervisor handoff",
            extra={
                "workflow_status": state.status,
                "planning_complete": state.planning_workflow_state.planning_complete,
                "next_phase": state.planning_workflow_state.next_phase
            }
        )
            
    def _handle_execution_planner_completion(self, state: PlannerSupervisorState, completion_data: list, agent_completion: dict) -> None:
        """
        Handle execution planner completion specifically.
        
        Args:
            state: The current planner supervisor state
            completion_data: The execution plan data (list of execution plans)
            agent_completion: The agent completion metadata
        """
        # Preserve existing execution data from previous state
        existing_data = self._planner_supervisor_state.execution_data
        if existing_data:
            state.execution_data.module_structure_plan = existing_data.module_structure_plan
            state.execution_data.module_structure_plan_complete = existing_data.module_structure_plan_complete
            state.execution_data.configuration_optimizer_data = existing_data.configuration_optimizer_data
            state.execution_data.configuration_optimizer_complete = existing_data.configuration_optimizer_complete
            state.execution_data.state_management_data = existing_data.state_management_data
            state.execution_data.state_management_complete = existing_data.state_management_complete
        
        # Store the new execution plan data and completion status
        # Wrap list in dict to match Pydantic model expectations
        if isinstance(completion_data, list):
            state.execution_data.execution_plan_data = {"execution_plans": completion_data}
        else:
            state.execution_data.execution_plan_data = completion_data
        state.execution_data.execution_plan_complete = True
        state.execution_data.agent_completion = agent_completion
        
        # CRITICAL: Mark phase complete using the proper method
        # This ensures both state objects are consistent and follows the same pattern as requirements
        state.set_phase_complete("execution_planning")
        
        # Log completion
        planner_supervisor_logger.log_structured(
            level="INFO",
            message="Execution planner completed",
            extra={
                "agent_name": agent_completion.get('agent_name'),
                "task_type": agent_completion.get('task_type'),
                "data_type": agent_completion.get('data_type'),
                "completion_timestamp": agent_completion.get('timestamp')
            }
        )
        
        # Mark phase complete
        state.set_phase_complete("execution_planning")
        state.execution_data.timestamp = datetime.now().isoformat()

    def _handle_requirements_analyzer_completion(self, state: PlannerSupervisorState, completion_data: dict, agent_completion: dict) -> None:
        """
        Handle requirements analyzer completion specifically.
        
        Args:
            state: The current planner supervisor state
            completion_data: The parsed completion data
            agent_completion: The agent completion metadata
        """
        # Store the new terraform attribute mapping data
        state.requirements_data.terraform_attribute_mapping = completion_data
        state.requirements_data.terraform_attribute_mapping_complete = True
        
        # Preserve existing requirements data from previous state
        existing_data = self._planner_supervisor_state.requirements_data
        if existing_data:
            state.requirements_data.aws_service_mapping = existing_data.aws_service_mapping
            state.requirements_data.aws_service_mapping_complete = existing_data.aws_service_mapping_complete
            state.requirements_data.analysis_results = existing_data.analysis_results
            state.requirements_data.analysis_complete = existing_data.analysis_complete
        
        # Log completion
        planner_supervisor_logger.log_structured(
            level="INFO",
            message="Requirements analyzer terraform attribute mapping completed",
            extra={
                "agent_name": agent_completion.get('agent_name'),
                "task_type": agent_completion.get('task_type'),
                "data_type": agent_completion.get('data_type'),
                "services_count": len(completion_data.get('services', [])),
                "total_resources": sum(len(service.get('terraform_resources', [])) for service in completion_data.get('services', [])),
                "completion_timestamp": agent_completion.get('timestamp')
            }
        )
        
        # Mark phase complete
        state.set_phase_complete("requirements_analysis")
        
        # Update timestamp
        state.requirements_data.timestamp = datetime.now().isoformat()

    def build_graph(self) -> StateGraph:
        """
        Build the LangGraph StateGraph for the planner sub-supervisor agent.
        
        Returns:
            StateGraph: The compiled graph for this agent
        """
        try:
            planner_supervisor_logger.log_structured(
                level="INFO",
                message="=== PLANNER SUB SUPERVISOR BUILD GRAPH START ===",
                extra={
                    "agent_name": getattr(self, '_name', 'unknown'),
                    "sub_agents_initialized": hasattr(self, 'requirements_analyzer') and hasattr(self, 'dependency_mapper') and hasattr(self, 'execution_planner'),
                    "handoff_tools_count": len(self.handoff_tools) if hasattr(self, 'handoff_tools') else 0,
                    "model_initialized": hasattr(self, 'model'),
                    "supervisor_prompt_defined": hasattr(self, 'supervisor_prompt'),
                    "memory_initialized": hasattr(self, 'memory')
                }
            )
            
            # Log sub-agents status
            planner_supervisor_logger.log_structured(
                level="DEBUG",
                message="Sub-agents status check",
                extra={
                    "requirements_analyzer_type": type(self.requirements_analyzer).__name__ if hasattr(self, 'requirements_analyzer') else "not_initialized",
                    "dependency_mapper_type": type(self.dependency_mapper).__name__ if hasattr(self, 'dependency_mapper') else "not_initialized",
                    "execution_planner_type": type(self.execution_planner).__name__ if hasattr(self, 'execution_planner') else "not_initialized"
                }
            )
            
            # Log handoff tools
            planner_supervisor_logger.log_structured(
                level="DEBUG",
                message="Handoff tools status",
                extra={
                    "handoff_tools_names": list(self.handoff_tools.keys()) if hasattr(self, 'handoff_tools') else [],
                    "handoff_tools_types": [type(tool).__name__ for tool in self.handoff_tools.values()] if hasattr(self, 'handoff_tools') else []
                }
            )
            
            # Create supervisor using langgraph-supervisor
            planner_supervisor_logger.log_structured(
                level="DEBUG",
                message="Creating supervisor with langgraph-supervisor",
                extra={
                    "agents_count": 3,
                    "state_schema": "PlannerSupervisorState",
                    "model_type": type(self.model).__name__ if hasattr(self, 'model') else "not_initialized"
                }
            )
            
            # Create pre-model hook to transform incoming state
            def pre_model_hook(state: PlannerSupervisorState) -> Dict[str, Any]:
                """Transform incoming state to properly extract user_request and other context."""
                try:
                    # Enhanced logging for cancellation detection
                    planner_supervisor_logger.log_structured(
                        level="DEBUG",
                        message="Pre-model hook: Transforming incoming state",
                        extra={
                            "state_type": type(state).__name__,
                            "messages_count": len(state.messages) if hasattr(state, 'messages') else 0,
                            "loop_counter": getattr(state.planning_workflow_state, 'loop_counter', 0) if hasattr(state, 'planning_workflow_state') else 0,
                            "current_phase": getattr(state.planning_workflow_state, 'current_phase', 'unknown') if hasattr(state, 'planning_workflow_state') else 'unknown',
                            "planning_complete": getattr(state.planning_workflow_state, 'planning_complete', False) if hasattr(state, 'planning_workflow_state') else False,
                            "execution_complete": getattr(state.planning_workflow_state, 'execution_complete', False) if hasattr(state, 'planning_workflow_state') else False,
                            "requirements_complete": getattr(state.planning_workflow_state, 'requirements_complete', False) if hasattr(state, 'planning_workflow_state') else False,
                            "timestamp": datetime.now().isoformat()
                        }
                    )
                    
                    # CRITICAL: Message content validation function (from research findings)
                    def is_valid_message_content(content):
                        """Validate message content according to strict LLM requirements."""
                        if not content:
                            return False
                        if isinstance(content, str):
                            return bool(content.strip())
                        if isinstance(content, list):
                            return any(
                                (isinstance(block, dict) and block.get('type') == 'text' and block.get('text', '').strip())
                                or (isinstance(block, str) and block.strip())
                                for block in content
                            )
                        return True
                    
                    # Initialize state updates dict (research-backed pattern)
                    state_updates = {}
                    
                    # CRITICAL: Ensure llm_input_messages is populated with VALID content
                    # This must be done FIRST before any other logic
                    if not state.llm_input_messages and state.messages:
                        valid_messages = []
                        for message in reversed(state.messages):
                            # Check if message has valid content
                            if is_valid_message_content(getattr(message, 'content', None)):
                                valid_messages.append(message)
                                planner_supervisor_logger.log_structured(
                                    level="DEBUG",
                                    message="Found valid message for llm_input_messages",
                                    extra={
                                        "message_type": type(message).__name__,
                                        "content_length": len(str(getattr(message, 'content', ''))),
                                        "content_preview": str(getattr(message, 'content', ''))[:100]
                                    }
                                )
                                break
                            # Patch tool call-only messages (from research findings)
                            elif hasattr(message, 'tool_calls') and message.tool_calls:
                                # Create a patched message with placeholder content
                                if isinstance(message, AIMessage):
                                    patched_message = AIMessage(
                                        content="[Tool execution completed]",
                                        tool_calls=message.tool_calls,
                                        additional_kwargs=getattr(message, 'additional_kwargs', {}),
                                        response_metadata=getattr(message, 'response_metadata', {}),
                                        id=getattr(message, 'id', None)
                                    )
                                    valid_messages.append(patched_message)
                                    planner_supervisor_logger.log_structured(
                                        level="DEBUG",
                                        message="Patched tool call-only message with placeholder content",
                                        extra={
                                            "original_message_type": type(message).__name__,
                                            "tool_calls_count": len(message.tool_calls)
                                        }
                                    )
                                    break
                        
                        if valid_messages:
                            state_updates["llm_input_messages"] = valid_messages
                            planner_supervisor_logger.log_structured(
                                level="INFO",
                                message="Successfully populated llm_input_messages with valid content",
                                extra={
                                    "valid_messages_count": len(valid_messages),
                                    "message_types": [type(msg).__name__ for msg in valid_messages]
                                }
                            )
                        else:
                            planner_supervisor_logger.log_structured(
                                level="WARNING",
                                message="No valid messages found for llm_input_messages",
                                extra={
                                    "total_messages": len(state.messages),
                                    "message_types": [type(msg).__name__ for msg in state.messages]
                                }
                            )
                    
                    # Check if this is a subsequent request (not first time)
                    is_subsequent_request = all([
                        hasattr(state, 'user_request') and state.user_request,
                        hasattr(state, 'task_description') and state.task_description,
                        hasattr(state, 'session_id') and state.session_id,
                        hasattr(state, 'task_id') and state.task_id
                    ])
                    
                    if is_subsequent_request:
                        # Process subsequent request - check for agent completions
                        self._process_subsequent_request(state)
                        
                        # CRITICAL: Don't overwrite llm_input_messages - only set if not already populated with valid content
                        if not state_updates.get("llm_input_messages"):
                            state_updates["llm_input_messages"] = [HumanMessage(content=state.user_request)]
                        
                    else:
                        # SECOND CONDITION: This IS the first request - extract user_request, task_description, etc.
                        planner_supervisor_logger.log_structured(
                            level="DEBUG",
                            message="Pre-model hook: Processing first request - extracting user_request, task_description, etc.",
                            extra={"messages_count": len(state.messages) if hasattr(state, 'messages') else 0}
                        )
                        
                        # Extract values directly from the PlannerSupervisorState messages
                        user_request = ""
                        task_description = ""
                        session_id = None
                        task_id = None
                        
                        if state.messages:
                            for message in state.messages:
                                # Handle HumanMessage
                                if isinstance(message, HumanMessage):
                                    if hasattr(message, 'content') and message.content:
                                        user_request = message.content
                                    
                                    if hasattr(message, 'additional_kwargs') and message.additional_kwargs:
                                        session_id = message.additional_kwargs.get('session_id', session_id)
                                        task_id = message.additional_kwargs.get('task_id', task_id)
                                
                                # Handle AIMessage
                                elif isinstance(message, AIMessage):
                                    if (hasattr(message, 'additional_kwargs') and 
                                        message.additional_kwargs and 
                                        'tool_calls' in message.additional_kwargs and
                                        message.additional_kwargs['tool_calls']):
                                        
                                        for tool_call in message.additional_kwargs['tool_calls']:
                                            if ('function' in tool_call and 
                                                'arguments' in tool_call['function'] and
                                                tool_call['function']['name'] == 'transfer_to_planner_sub_supervisor'):
                                                
                                                try:
                                                    args = json.loads(tool_call['function']['arguments'])
                                                    task_description = args.get('task_description', '')
                                                except json.JSONDecodeError:
                                                    task_description = tool_call['function']['arguments']
                                                break
                                
                                # Handle SystemMessage (if needed)
                                elif isinstance(message, SystemMessage):
                                    # System messages typically don't contain user data
                                    pass
                        
                        planner_supervisor_logger.log_structured(
                            level="INFO",
                            message="Pre-model hook: Extracted values from first request",
                            extra={
                                "extracted_user_request": user_request,
                                "extracted_task_description": task_description,
                                "extracted_session_id": session_id,
                                "extracted_task_id": task_id
                            }
                        )
                        
                        # CRITICAL: Update state via state_updates dict (research-backed pattern)
                        # NO in-place mutations - LangGraph ignores them
                        state_updates.update({
                            "user_request": user_request,
                            "task_description": task_description,
                            "session_id": session_id,
                            "task_id": task_id
                        })
                        
                        # CRITICAL: Only set llm_input_messages if not already populated with valid content
                        if not state_updates.get("llm_input_messages"):
                            state_updates["llm_input_messages"] = [HumanMessage(content=user_request)]
                        
                        planner_supervisor_logger.log_structured(
                            level="INFO",
                            message="Pre-model hook: Updated state via state_updates dict",
                            extra={
                                "user_request": user_request,
                                "task_description": task_description,
                                "session_id": session_id,
                                "task_id": task_id,
                                "state_updates_keys": list(state_updates.keys())
                            }
                        )
                        
                        # Create llm_input_messages with the extracted user request
                        # state_updates["llm_input_messages"] = [HumanMessage(content=user_request)]
                        self._planner_supervisor_state.session_id = state.session_id
                        self._planner_supervisor_state.task_id = state.task_id
                        self._planner_supervisor_state.user_request = state.user_request
                        self._planner_supervisor_state.task_description = state.task_description
                        self._planner_supervisor_state.planning_workflow_state.current_phase = state.planning_workflow_state.current_phase
                    
                    
                    # CRITICAL: Return state updates dict (research-backed pattern)
                    # For message updates, we need to use the RemoveMessage pattern
                    if "llm_input_messages" in state_updates:
                        from langchain_core.messages import RemoveMessage
                        from langgraph.graph.message import REMOVE_ALL_MESSAGES
                        
                        # Use the correct pattern for message overwrites
                        new_llm_messages = state_updates["llm_input_messages"]
                        state_updates["llm_input_messages"] = [
                            RemoveMessage(id=REMOVE_ALL_MESSAGES), 
                            *new_llm_messages
                        ]
                    
                    planner_supervisor_logger.log_structured(
                        level="DEBUG",
                        message="Pre-model hook: Returning state updates with RemoveMessage pattern",
                        extra={
                            "state_updates_keys": list(state_updates.keys()),
                            "llm_input_messages_count": len(state_updates.get("llm_input_messages", [])),
                            "messages_count": len(state_updates.get("messages", [])),
                            "is_subsequent_request": is_subsequent_request
                        }
                    )
                    return state_updates
                    
                except Exception as e:
                    planner_supervisor_logger.log_structured(
                        level="ERROR",
                        message="Pre-model hook: Error transforming state",
                        extra={
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "traceback": traceback.format_exc()
                        }
                    )
                    # Return empty dict on error (research-backed pattern)
                    return {}
            
            # Create supervisor with the SAME state instance that requirements analyzer uses
            # Add proper configuration to prevent CancelledError issues
            planner_supervisor = create_supervisor(
                agents=[self.requirements_analyzer, self.execution_planner],
                prompt=self.supervisor_prompt,
                model=self.model,
                tools=list(self.handoff_tools.values()),
                state_schema=PlannerSupervisorState,
                pre_model_hook=pre_model_hook,
                add_handoff_back_messages=False,  # Ensure proper message handling
                output_mode="last_message",
                parallel_tool_calls=False,
                # output_mode="full_history"  # Enable full history for debugging
            )
            
            # Debug: Log tool availability
            planner_supervisor_logger.log_structured(
                level="DEBUG",
                message="Handoff tools registered with supervisor",
                extra={
                    "tools_count": len(self.handoff_tools),
                    "tool_names": list(self.handoff_tools.keys()),
                    "tool_types": [type(tool).__name__ for tool in self.handoff_tools.values()],
                    "has_handoff_to_planner_complete": "handoff_to_planner_complete" in self.handoff_tools
                }
            )
            
            planner_supervisor_logger.log_structured(
                level="INFO",
                message="=== PLANNER SUB SUPERVISOR BUILD GRAPH COMPLETE ===",
                extra={
                    "supervisor_type": type(planner_supervisor).__name__,
                    "agents_count": 3,
                    "handoff_tools_count": len(self.handoff_tools),
                    "state_schema": "PlannerSupervisorState",
                    "supervisor_has_nodes": hasattr(planner_supervisor, 'nodes'),
                    "supervisor_nodes": list(planner_supervisor.nodes.keys()) if hasattr(planner_supervisor, 'nodes') else "unknown"
                }
            )
            
            return planner_supervisor
            
        except Exception as e:
            planner_supervisor_logger.log_structured(
                level="ERROR",
                message="=== PLANNER SUB SUPERVISOR BUILD GRAPH FAILED ===",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc()
                }
            )
            raise
    
    def input_transform(self, send_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform Send() payload from supervisor to agent state.
        
        Args:
            send_payload: Data sent from supervisor via Send() primitive
            
        Returns:
            Dict[str, Any]: Transformed state ready for agent processing
        """
        try:
            planner_supervisor_logger.log_structured(
                level="INFO",
                message="=== PLANNER SUB SUPERVISOR INPUT TRANSFORM START ===",
                extra={
                    "send_payload_type": type(send_payload).__name__,
                    "send_payload_keys": list(send_payload.keys()) if isinstance(send_payload, dict) else "not_dict",
                    "send_payload_size": len(str(send_payload)) if isinstance(send_payload, dict) else 0
                }
            )
            
            # Extract task description and user request from Send payload
            # The supervisor handoff tools pass these directly in the payload
            task_description = send_payload.get("task_description", "")
            user_request = send_payload.get("user_request", task_description)
            
            planner_supervisor_logger.log_structured(
                level="DEBUG",
                message="Extracted initial values from send_payload",
                extra={
                    "task_description_length": len(task_description),
                    "user_request_length": len(user_request),
                    "task_description_preview": task_description[:100] + "..." if len(task_description) > 100 else task_description,
                    "user_request_preview": user_request[:100] + "..." if len(user_request) > 100 else user_request
                }
            )
            
            # If both are empty, try to extract from messages as fallback
            if not task_description and not user_request:
                planner_supervisor_logger.log_structured(
                    level="WARNING",
                    message="Both task_description and user_request are empty, trying fallback extraction",
                    extra={"has_messages": "messages" in send_payload}
                )
                
                if "messages" in send_payload and send_payload["messages"]:
                    planner_supervisor_logger.log_structured(
                        level="DEBUG",
                        message="Attempting to extract from messages",
                        extra={"messages_count": len(send_payload["messages"])}
                    )
                    
                    # Get the last human message content
                    for i, message in enumerate(reversed(send_payload["messages"])):
                        planner_supervisor_logger.log_structured(
                            level="DEBUG",
                            message=f"Checking message {i}",
                            extra={
                                "message_type": type(message).__name__,
                                "has_content": hasattr(message, 'content'),
                                "content_length": len(message.content) if hasattr(message, 'content') and message.content else 0,
                                "has_additional_kwargs": hasattr(message, 'additional_kwargs'),
                                "additional_kwargs_keys": list(message.additional_kwargs.keys()) if hasattr(message, 'additional_kwargs') else []
                            }
                        )
                        
                        if hasattr(message, 'content') and message.content:
                            # First try to get user_request from additional_kwargs
                            if hasattr(message, 'additional_kwargs') and message.additional_kwargs:
                                user_request = message.additional_kwargs.get('user_request', message.content)
                                task_description = message.additional_kwargs.get('task_description', message.content)
                            else:
                                # Fallback to content
                                task_description = message.content
                                user_request = message.content
                            
                            planner_supervisor_logger.log_structured(
                                level="INFO",
                                message="Successfully extracted from message",
                                extra={
                                    "extracted_task_description_length": len(task_description),
                                    "extracted_user_request_length": len(user_request),
                                    "extracted_task_description_preview": task_description[:100] + "..." if len(task_description) > 100 else task_description,
                                    "extracted_user_request_preview": user_request[:100] + "..." if len(user_request) > 100 else user_request,
                                    "source": "additional_kwargs" if hasattr(message, 'additional_kwargs') and message.additional_kwargs else "content"
                                }
                            )
                            break
            
            # Ensure we have some content
            if not task_description and not user_request:
                planner_supervisor_logger.log_structured(
                    level="WARNING",
                    message="No content found, using default placeholder",
                    extra={"original_task_description": task_description, "original_user_request": user_request}
                )
                task_description = "User request not provided"
                user_request = "User request not provided"
            
            planner_supervisor_logger.log_structured(
                level="DEBUG",
                message="Final extracted values",
                extra={
                    "final_task_description_length": len(task_description),
                    "final_user_request_length": len(user_request),
                    "final_task_description_preview": task_description[:100] + "..." if len(task_description) > 100 else task_description,
                    "final_user_request_preview": user_request[:100] + "..." if len(user_request) > 100 else user_request
                }
            )
            
            # Create initial planner state
            planner_supervisor_logger.log_structured(
                level="DEBUG",
                message="Creating initial planner state",
                extra={
                    "session_id": send_payload.get("session_id"),
                    "task_id": send_payload.get("task_id")
                }
            )
            
            initial_state = create_initial_planner_state(
                user_request=user_request,
                session_id=send_payload.get("session_id"),
                task_id=send_payload.get("task_id")
            )
            
            planner_supervisor_logger.log_structured(
                level="DEBUG",
                message="Initial state created successfully",
                extra={
                    "initial_state_user_request": initial_state.user_request,
                    "initial_state_session_id": initial_state.session_id,
                    "initial_state_task_id": initial_state.task_id
                }
            )
            
            # Add initial message
            # from langchain_core.messages import HumanMessage
            initial_state.messages = [HumanMessage(content=task_description)]
            
            planner_supervisor_logger.log_structured(
                level="DEBUG",
                message="Added initial message to state",
                extra={"initial_message_content_length": len(task_description)}
            )
            
            # Transform to agent's state model format
            transformed_state = {
                "messages": initial_state.messages,
                "user_request": initial_state.user_request,
                "session_id": initial_state.session_id,
                "task_id": initial_state.task_id,
                "remaining_steps": initial_state.remaining_steps,
                "planning_workflow_state": initial_state.planning_workflow_state,
                "requirements_data": initial_state.requirements_data,
                "dependency_data": initial_state.dependency_data,
                "execution_data": initial_state.execution_data,
                "planning_results": initial_state.planning_results,
                "active_agent": initial_state.active_agent,
                "task_description": initial_state.task_description,
                "planning_context": initial_state.planning_context,
                "status": initial_state.status,
                "error": initial_state.error,
                "metadata": initial_state.metadata,
                "supervisor_context": send_payload.get("context", {}),
            }
            
            planner_supervisor_logger.log_structured(
                level="INFO",
                message="=== PLANNER SUB SUPERVISOR INPUT TRANSFORM COMPLETE ===",
                extra={
                    "transformed_state_keys": list(transformed_state.keys()),
                    "final_user_request": transformed_state["user_request"],
                    "final_task_description": transformed_state["task_description"],
                    "workflow_state_phase": transformed_state["planning_workflow_state"].current_phase if hasattr(transformed_state["planning_workflow_state"], 'current_phase') else "unknown"
                }
            )
            
            return transformed_state
            
        except Exception as e:
            planner_supervisor_logger.log_structured(
                level="ERROR",
                message=f"=== PLANNER SUB SUPERVISOR INPUT TRANSFORM FAILED ===",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "send_payload_keys": list(send_payload.keys()) if isinstance(send_payload, dict) else "not_dict",
                    "traceback": traceback.format_exc()
                }
            )
            raise
    
    def output_transform(self, agent_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform agent state back to supervisor state.
        
        Args:
            agent_state: The final state from agent execution
        
        Returns:
            Dict[str, Any]: Data to merge into supervisor state
        """
        try:
            planner_supervisor_logger.log_structured(
                level="INFO",
                message="=== PLANNER SUB SUPERVISOR OUTPUT TRANSFORM START ===",
                extra={
                    "agent_state_type": type(agent_state).__name__,
                    "agent_state_keys": list(agent_state.keys()) if isinstance(agent_state, dict) else "not_dict",
                    "agent_state_size": len(str(agent_state)) if isinstance(agent_state, dict) else 0
                }
            )
            
            # Extract planning results
            planning_results = agent_state.get("planning_results")
            
            planner_supervisor_logger.log_structured(
                level="DEBUG",
                message="Planning results extraction",
                extra={
                    "has_planning_results": planning_results is not None,
                    "planning_results_type": type(planning_results).__name__ if planning_results is not None else "None"
                }
            )
            
            if not planning_results:
                # Create planning results from state if not present
                planner_supervisor_logger.log_structured(
                    level="DEBUG",
                    message="Creating planning results from state",
                    extra={}
                )
                
                planning_results = create_planning_results(agent_state)
                
                planner_supervisor_logger.log_structured(
                    level="DEBUG",
                    message="Planning results created",
                    extra={
                        "planning_results_type": type(planning_results).__name__
                    }
                )
            
            # Prepare output
            output_data = {
                "messages": agent_state.get("messages", []),
                "agent_result": {
                    "planning_results": planning_results,
                    "planning_workflow_state": agent_state.get("planning_workflow_state"),
                    "requirements_data": agent_state.get("requirements_data"),
                    "dependency_data": agent_state.get("dependency_data"),
                    "execution_data": agent_state.get("execution_data"),
                },
                "agent_status": agent_state.get("status", "completed"),
                "agent_metadata": {
                    "agent_name": self.name,
                    "planning_phase": agent_state.get("planning_workflow_state", {}).get("current_phase"),
                    "complexity_score": getattr(planning_results, 'overall_complexity_score', None),
                    "risk_level": getattr(planning_results, 'risk_level', None),
                }
            }
            
            planner_supervisor_logger.log_structured(
                level="INFO",
                message="=== PLANNER SUB SUPERVISOR OUTPUT TRANSFORM COMPLETE ===",
                extra={
                    "output_keys": list(output_data.keys()),
                    "agent_status": output_data["agent_status"],
                    "planning_phase": output_data["agent_metadata"]["planning_phase"],
                    "messages_count": len(output_data["messages"])
                }
            )
            
            return output_data
            
        except Exception as e:
            planner_supervisor_logger.log_structured(
                level="ERROR",
                message="=== PLANNER SUB SUPERVISOR OUTPUT TRANSFORM FAILED ===",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "agent_state_keys": list(agent_state.keys()) if isinstance(agent_state, dict) else "not_dict",
                    "traceback": traceback.format_exc()
                }
            )
            raise
    
    def can_interrupt(self) -> bool:
        """
        Whether this agent can interrupt for human input.
        
        Returns:
            bool: True if agent can interrupt, False otherwise
        """
        return False  # Planning workflow does not require human input


# Factory function for easy creation
@log_sync
def create_planner_sub_supervisor_agent(
    config: Optional[Config] = None,
    custom_config: Optional[Dict[str, Any]] = None,
    name: str = "planner_sub_supervisor"
) -> PlannerSubSupervisorAgent:
    """
    Factory function to create a Planner Sub-Supervisor Agent.
    
    Args:
        config: Configuration instance
        custom_config: Optional custom configuration
        name: Agent name
        
    Returns:
        Configured PlannerSubSupervisorAgent instance
    """
    return PlannerSubSupervisorAgent(config=config, custom_config=custom_config, name=name)


# Backward compatibility function
def create_planner_sub_supervisor_agent_factory(config: Config):
    """
    Factory function for creating planner sub-supervisor agent.
    Maintains backward compatibility with existing code.
    
    Args:
        config: Configuration object
        
    Returns:
        Configured PlannerSubSupervisorAgent instance
    """
    return create_planner_sub_supervisor_agent(config=config)
