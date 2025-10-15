"""
Task Lifecycle Manager for A2A Protocol.

This module provides the central TaskLifecycleManager class that handles
task state transitions, history tracking, and integration with a2a-sdk components.
"""

import logging
from datetime import datetime, timedelta, UTC
from typing import Dict, List, Optional, Any
from uuid import uuid4

from a2a.server.tasks import TaskStore
from a2a.server.events import EventQueue

from aws_orchestrator_agent.schemas.task_lifecycle import (
    TaskState,
    TaskTransition,
    TaskStatus,
    TaskArtifact,
    TaskHistory,
    TaskLifecycleConfig,
    TaskMetrics,
)

logger = logging.getLogger(__name__)


class TaskLifecycleManager:
    """
    Central manager for task lifecycle operations.
    
    This class provides comprehensive task lifecycle management including:
    - State transition validation and enforcement
    - Task history tracking with audit trail
    - Integration with a2a-sdk TaskStore and EventQueue
    - Metrics collection and monitoring
    - Artifact management
    """
    
    def __init__(
        self,
        task_store: Optional[TaskStore] = None,
        event_queue: Optional[EventQueue] = None,
        config: Optional[TaskLifecycleConfig] = None
    ):
        """
        Initialize the TaskLifecycleManager.
        
        Args:
            task_store: Optional a2a-sdk TaskStore for persistent storage
            event_queue: Optional a2a-sdk EventQueue for event propagation
            config: Configuration for lifecycle management
        """
        self.task_store = task_store
        self.event_queue = event_queue
        self.config = config or TaskLifecycleConfig()
        
        # In-memory storage for active tasks and history
        self._active_tasks: Dict[str, TaskHistory] = {}
        self._task_metrics: Dict[str, TaskMetrics] = {}
        
        # State transition validation rules (A2A Protocol compliant)
        self._valid_transitions = {
            TaskState.SUBMITTED: [TaskState.WORKING, TaskState.CANCELED, TaskState.FAILED],
            TaskState.WORKING: [TaskState.INPUT_REQUIRED, TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED],
            TaskState.INPUT_REQUIRED: [TaskState.WORKING, TaskState.FAILED, TaskState.CANCELED],
            TaskState.COMPLETED: [],  # Terminal state
            TaskState.FAILED: [],     # Terminal state
            TaskState.CANCELED: [],   # Terminal state
        }
        
        logger.info("TaskLifecycleManager initialized")
    
    async def create_task(
        self,
        task_id: str,
        context_id: str,
        initial_state: TaskState = TaskState.SUBMITTED,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TaskHistory:
        logger.debug(f"[TaskLifecycleManager] create_task called with task_id={task_id}, context_id={context_id}, initial_state={initial_state}")
        if task_id in self._active_tasks:
            logger.warning(f"[TaskLifecycleManager] Task {task_id} already exists")
            raise ValueError(f"Task {task_id} already exists")
        
        # Create task history
        task_history = TaskHistory(
            task_id=task_id,
            created_at=datetime.now(UTC),
            transitions=[],
            events=[],
            artifacts=[]
        )
        
        # Add initial transition - only create transition if initial_state is different from SUBMITTED
        if initial_state != TaskState.SUBMITTED:
            initial_transition = TaskTransition(
                from_state=TaskState.SUBMITTED,  # Always start from SUBMITTED
                to_state=initial_state,
                timestamp=datetime.now(UTC),
                reason="Task created",
                metadata=metadata or {},
                agent_id=None,
                user_id=None
            )
            task_history.add_transition(initial_transition)
        else:
            # For SUBMITTED initial state, create a transition from None to SUBMITTED
            initial_transition = TaskTransition(
                from_state=None,  # No previous state
                to_state=TaskState.SUBMITTED,
                timestamp=datetime.now(UTC),
                reason="Task created",
                metadata=metadata or {},
                agent_id=None,
                user_id=None
            )
            task_history.add_transition(initial_transition)
        
        # Store in memory
        self._active_tasks[task_id] = task_history
        
        # Initialize metrics
        self._task_metrics[task_id] = TaskMetrics(
            task_id=task_id,
            total_duration_seconds=0.0,
            processing_duration_seconds=0.0,
            waiting_duration_seconds=0.0,
            transition_count=1,
            retry_count=0,
            artifact_count=0,
            success=False
        )
        
        # Store in persistent storage if available
        if self.task_store:
            await self._persist_task(task_history)
        
        # Emit event if available
        if self.event_queue:
            await self._emit_task_event(task_id, "task_created", {
                "task_id": task_id,
                "context_id": context_id,
                "initial_state": initial_state.value,
                "metadata": metadata
            })
        
        logger.debug(f"[TaskLifecycleManager] Created task: {task_id} with state {initial_state}")
        return task_history
    
    async def transition_task_state(
        self,
        task_id: str,
        new_state: TaskState,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> TaskTransition:
        logger.debug(f"[TaskLifecycleManager] transition_task_state called for task_id={task_id}, new_state={new_state}")
        if task_id not in self._active_tasks:
            raise ValueError(f"Task {task_id} not found")
        
        task_history = self._active_tasks[task_id]
        current_state = task_history.current_state
        
        if current_state is None:
            raise ValueError(f"Task {task_id} has no current state")
        
        # Validate transition
        if not self._is_valid_transition(current_state, new_state):
            raise ValueError(
                f"Invalid transition from {current_state.value} to {new_state.value}"
            )
        
        # Create transition
        transition = TaskTransition(
            from_state=current_state,
            to_state=new_state,
            timestamp=datetime.now(UTC),
            reason=reason or f"State changed from {current_state.value} to {new_state.value}",
            metadata=metadata or {},
            agent_id=agent_id,
            user_id=user_id
        )
        
        # Add to history
        task_history.add_transition(transition)
        
        # Update metrics
        await self._update_task_metrics(task_id, transition)
        
        # Store in persistent storage if available
        if self.task_store:
            await self._persist_task(task_history)
        
        # Emit event if available
        if self.event_queue:
            await self._emit_task_event(task_id, "task_state_changed", {
                "task_id": task_id,
                "from_state": current_state.value,
                "to_state": new_state.value,
                "reason": reason,
                "agent_id": agent_id,
                "user_id": user_id
            })
        
        logger.debug(f"[TaskLifecycleManager] transition_task_state result: {transition}")
        return transition
    
    async def add_task_artifact(
        self,
        task_id: str,
        name: str,
        artifact_type: str,
        content: Optional[str] = None,
        file_path: Optional[str] = None,
        mime_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TaskArtifact:
        logger.debug(f"[TaskLifecycleManager] add_task_artifact called for task_id={task_id}, name={name}, type={artifact_type}")
        if task_id not in self._active_tasks:
            raise ValueError(f"Task {task_id} not found")
        
        # Create artifact
        artifact = TaskArtifact(
            id=str(uuid4()),
            name=name,
            type=artifact_type,
            content=content,
            file_path=file_path,
            mime_type=mime_type,
            metadata=metadata or {}
        )
        
        # Add to task history
        task_history = self._active_tasks[task_id]
        task_history.add_artifact(artifact)
        
        # Update metrics
        if task_id in self._task_metrics:
            self._task_metrics[task_id].artifact_count += 1
        
        # Store in persistent storage if available
        if self.task_store:
            await self._persist_task(task_history)
        
        # Emit event if available
        if self.event_queue:
            await self._emit_task_event(task_id, "task_artifact_added", {
                "task_id": task_id,
                "artifact_id": artifact.id,
                "artifact_name": name,
                "artifact_type": artifact_type
            })
        
        logger.debug(f"[TaskLifecycleManager] add_task_artifact result: {artifact}")
        return artifact
    
    async def get_task_history(self, task_id: str) -> Optional[TaskHistory]:
        logger.debug(f"[TaskLifecycleManager] get_task_history called for task_id={task_id}")
        result = self._active_tasks.get(task_id)
        logger.debug(f"[TaskLifecycleManager] get_task_history result: {result}")
        return result
    
    async def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """
        Get the current status of a task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            TaskStatus instance or None if not found
        """
        task_history = await self.get_task_history(task_id)
        if not task_history or not task_history.transitions:
            return None
        
        latest_transition = task_history.transitions[-1]
        
        return TaskStatus(
            state=latest_transition.to_state,
            message=latest_transition.reason,
            timestamp=latest_transition.timestamp,
            metadata=latest_transition.metadata
        )
    
    async def get_task_metrics(self, task_id: str) -> Optional[TaskMetrics]:
        logger.debug(f"[TaskLifecycleManager] get_task_metrics called for task_id={task_id}")
        result = self._task_metrics.get(task_id)
        logger.debug(f"[TaskLifecycleManager] get_task_metrics result: {result}")
        return result
    
    async def list_active_tasks(self, state_filter: Optional[TaskState] = None) -> List[str]:
        """
        List active task IDs, optionally filtered by state.
        
        Args:
            state_filter: Optional state filter
            
        Returns:
            List of task IDs
        """
        if state_filter is None:
            return list(self._active_tasks.keys())
        
        return [
            task_id for task_id, history in self._active_tasks.items()
            if history.current_state == state_filter
        ]
    
    async def cleanup_completed_tasks(self) -> int:
        """
        Clean up completed tasks based on configuration.
        
        Returns:
            Number of tasks cleaned up
        """
        if not self.config.auto_cleanup_completed_tasks:
            return 0
        
        cutoff_time = datetime.now(UTC) - timedelta(days=self.config.cleanup_delay_days)
        cleaned_count = 0
        
        tasks_to_cleanup = []
        for task_id, history in self._active_tasks.items():
            if history.current_state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED]:
                if history.transitions and history.transitions[-1].timestamp < cutoff_time:
                    tasks_to_cleanup.append(task_id)
        
        for task_id in tasks_to_cleanup:
            await self._cleanup_task(task_id)
            cleaned_count += 1
        
        logger.info(f"Cleaned up {cleaned_count} completed tasks")
        return cleaned_count
    
    def _is_valid_transition(self, from_state: TaskState, to_state: TaskState) -> bool:
        """
        Check if a state transition is valid.
        
        Args:
            from_state: Current state
            to_state: Target state
            
        Returns:
            True if transition is valid
        """
        allowed_transitions = self._valid_transitions.get(from_state, [])
        return to_state in allowed_transitions
    
    async def _update_task_metrics(self, task_id: str, transition: TaskTransition) -> None:
        """
        Update task metrics based on a state transition.
        
        Args:
            task_id: Task identifier
            transition: The state transition
        """
        if task_id not in self._task_metrics:
            return
        
        metrics = self._task_metrics[task_id]
        metrics.transition_count += 1
        
        # Get task history to calculate durations
        task_history = self._active_tasks.get(task_id)
        if not task_history or len(task_history.transitions) < 2:
            return
        
        # Calculate duration for the previous state
        prev_transition = task_history.transitions[-2]  # Previous transition
        duration = (transition.timestamp - prev_transition.timestamp).total_seconds()
        
        if prev_transition.to_state == TaskState.WORKING:
            metrics.processing_duration_seconds += duration
        elif prev_transition.to_state == TaskState.INPUT_REQUIRED:
            metrics.waiting_duration_seconds += duration
        
        # Update success flag for terminal states
        if transition.to_state == TaskState.COMPLETED:
            metrics.success = True
        elif transition.to_state in [TaskState.FAILED, TaskState.CANCELED]:
            metrics.success = False
        
        # Calculate total duration
        if len(task_history.transitions) > 1:
            first_transition = task_history.transitions[0]
            metrics.total_duration_seconds = (
                transition.timestamp - first_transition.timestamp
            ).total_seconds()
    
    async def _persist_task(self, task_history: TaskHistory) -> None:
        """
        Persist task history to storage.
        
        Args:
            task_history: Task history to persist
        """
        if not self.task_store:
            return
        
        try:
            # Convert to a2a-sdk Task format if needed
            # This is a simplified implementation - you may need to adapt based on your a2a-sdk version
            await self.task_store.store_task(task_history.task_id, task_history.dict())
        except Exception as e:
            logger.error(f"Failed to persist task {task_history.task_id}: {e}")
    
    async def _emit_task_event(self, task_id: str, event_type: str, data: Dict[str, Any]) -> None:
        """
        Emit a task-related event.
        
        Args:
            task_id: Task identifier
            event_type: Type of event
            data: Event data
        """
        if not self.event_queue:
            return
        
        try:
            event_data = {
                "task_id": task_id,
                "event_type": event_type,
                "timestamp": datetime.now(UTC).isoformat(),
                **data
            }
            await self.event_queue.enqueue_event(event_data)
        except Exception as e:
            logger.error(f"Failed to emit event for task {task_id}: {e}")
    
    async def _cleanup_task(self, task_id: str) -> None:
        """
        Clean up a specific task.
        
        Args:
            task_id: Task identifier to cleanup
        """
        if task_id in self._active_tasks:
            del self._active_tasks[task_id]
        
        if task_id in self._task_metrics:
            del self._task_metrics[task_id]
        
        logger.debug(f"Cleaned up task {task_id}") 