from aws_orchestrator_agent.utils.logger import AgentLogger
import inspect
import abc

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    DataPart,
    InvalidParamsError,
    SendStreamingMessageSuccessResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
    TextPart,
    UnsupportedOperationError,
    Part,
)
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError
from aws_orchestrator_agent.types import BaseAgent
from typing import cast, Any


logger = AgentLogger("GENERIC_AGENT_EXECUTOR")


class ExecutorValidationMixin(abc.ABC):
    """
    Mixin to enforce extra validation/status mapping methods for all custom executors.
    """
    @abc.abstractmethod
    def _validate_request(self, context: RequestContext) -> bool:
        """
        Validate the incoming request context. Return True if invalid, False if valid.
        """
        pass

    @abc.abstractmethod
    def _map_status_to_task_state(self, custom_status: str) -> TaskState:
        """
        Map custom status strings to A2A TaskState enum values.
        """
        pass


class GenericAgentExecutor(AgentExecutor, ExecutorValidationMixin):
    """AgentExecutor used by the tragel agents with JSON-RPC 2.0 validation support."""

    def __init__(self, agent: BaseAgent) -> None:
        self.agent: BaseAgent = agent

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        logger.log_structured(
            level="INFO",
            message=f'Executing agent {self.agent.name}',
            extra={"agent_name": self.agent.name}
        )
        error = self._validate_request(context)
        if error:
            logger.log_structured(
                level="ERROR",
                message='Validation error in request context',
                extra={"agent_name": self.agent.name}
            )
            raise ServerError(error=InvalidParamsError())

        query = context.get_user_input()
        logger.log_structured(
            level="DEBUG",
            message=f'User query: {query}',
            extra={"agent_name": self.agent.name}
        )

        task = context.current_task

        if not task:
            logger.log_structured(
                level="INFO",
                message='No current task found, creating new task',
                extra={"agent_name": self.agent.name}
            )
            if context.message is None:
                logger.log_structured(
                    level="ERROR",
                    message='No message provided for new task',
                    extra={"agent_name": self.agent.name}
                )
                raise ServerError(error=InvalidParamsError())
            task = new_task(context.message)
            logger.log_structured(
                level="INFO",
                message=f'Created new task with id: {task.id}, contextId: {task.contextId}',
                extra={"agent_name": self.agent.name, "task_id": task.id, "context_id": task.contextId}
            )
            await event_queue.enqueue_event(task)
            logger.log_structured(
                level="DEBUG",
                message=f'Enqueued new task: {task}',
                extra={"agent_name": self.agent.name, "task_id": task.id, "context_id": task.contextId}
            )

        updater = TaskUpdater(event_queue, task.id, task.contextId)
        logger.log_structured(
            level="INFO",
            message=f'Starting agent stream',
            extra={"agent_name": self.agent.name, "task_id": task.id, "context_id": task.contextId}
        )

        try:
            # Ensure self.agent.stream is an async generator, or await if it's a coroutine returning one
            agent_stream = self.agent.stream(query, task.contextId, task.id)
            if not inspect.isasyncgen(agent_stream):
                agent_stream = await agent_stream  # type: ignore
            async for item in agent_stream:  # type: ignore
                # logger.debug(f'Received item from agent stream: {item}')
                # Forward agent-to-agent events directly to the event queue
                root = getattr(item, 'root', None)
                if root is not None and isinstance(root, SendStreamingMessageSuccessResponse):
                    event = root.result
                    if isinstance(
                        event,
                        (TaskStatusUpdateEvent, TaskArtifactUpdateEvent),
                    ):
                        logger.log_structured(
                            level="INFO",
                            message=f'Enqueuing event from agent: {event}',
                            extra={"agent_name": self.agent.name, "task_id": task.id, "context_id": task.contextId}
                        )
                        await event_queue.enqueue_event(event)
                    continue
                
                is_task_complete = item.is_task_complete
                require_user_input = item.require_user_input
                # logger.debug(f'is_task_complete={is_task_complete}, require_user_input={require_user_input}')
                
                # Map custom status to A2A TaskState enum
                custom_status = getattr(item, 'metadata', {}).get('status', 'working')
                task_state = self._map_status_to_task_state(custom_status)
                # logger.debug(f'Mapped custom status "{custom_status}" to task_state {task_state}')

                if is_task_complete:
                    logger.log_structured(
                        level="INFO",
                        message='Task is marked as complete by agent',
                        extra={"agent_name": self.agent.name, "task_id": task.id, "context_id": task.contextId}
                    )
                    if item.response_type == 'data':
                        data_part: Part = cast(Part, DataPart(data=item.content))
                    else:
                        text_part: Part = cast(Part, TextPart(text=item.content))

                    logger.log_structured(
                        level="INFO",
                        message='Adding artifact to updater',
                        extra={"agent_name": self.agent.name, "task_id": task.id, "context_id": task.contextId}
                    )
                    if item.response_type == 'data':
                        await updater.add_artifact(
                            [data_part],
                            name=f'{self.agent.name}-result',
                        )
                    else:
                        await updater.add_artifact(
                            [text_part],
                            name=f'{self.agent.name}-result',
                        )
                    logger.log_structured(
                        level="INFO",
                        message='Sending final status update: TaskState.completed',
                        extra={"agent_name": self.agent.name, "task_id": task.id, "context_id": task.contextId}
                    )
                    await updater.update_status(
                        TaskState.completed,
                        new_agent_text_message(
                            "Task completed successfully.",
                            task.contextId,
                            task.id,
                        ),
                        final=True,
                    )
                    logger.log_structured(
                        level="INFO",
                        message='Calling updater.complete()',
                        extra={"agent_name": self.agent.name, "task_id": task.id, "context_id": task.contextId}
                    )
                    await updater.complete()
                    logger.log_structured(
                        level="INFO",
                        message='Updater.complete() finished, breaking stream loop',
                        extra={"agent_name": self.agent.name, "task_id": task.id, "context_id": task.contextId}
                    )
                    break
                if require_user_input:
                    logger.log_structured(
                        level="INFO",
                        message='Agent requires user input, updating status to input_required',
                        extra={"agent_name": self.agent.name, "task_id": task.id, "context_id": task.contextId}
                    )
                    await updater.update_status(
                        TaskState.input_required,
                        new_agent_text_message(
                            item.content,
                            task.contextId,
                            task.id,
                        ),
                        final=True,
                    )
                    logger.log_structured(
                        level="INFO",
                        message='Status updated to input_required, breaking stream loop',
                        extra={"agent_name": self.agent.name, "task_id": task.id, "context_id": task.contextId}
                    )
                    break
                logger.log_structured(
                    level="INFO",
                    message=f'Updating status to {task_state}',
                    extra={"agent_name": self.agent.name, "task_id": task.id, "context_id": task.contextId}
                )
                await updater.update_status(
                    task_state,
                    new_agent_text_message(
                        item.content,
                        task.contextId,
                        task.id,
                    ),
                )
                # logger.debug('Status update sent')
        except Exception as e:
            logger.log_structured(
                level="ERROR",
                message=f'Exception in agent executor stream: {e}',
                extra={"agent_name": self.agent.name}
            )
            raise

    async def cancel(self, request: RequestContext, event_queue: EventQueue) -> Task | None:
        """
        Cancel the current agent execution if possible. Default implementation returns None.
        """
        return None

    def _validate_request(self, context: RequestContext) -> bool:
        """
        Validate the incoming request context. Default implementation returns False (valid).
        """
        return False

    def _map_status_to_task_state(self, custom_status: str) -> TaskState:
        """
        Map custom status strings to A2A TaskState enum values. Can be overridden by subclasses.
        """
        status_mapping: dict[str, TaskState] = {
            'working': TaskState.working,
            'input_required': TaskState.input_required,
            'completed': TaskState.completed,
            'failed': TaskState.failed,
            'error': TaskState.failed,
            'submitted': TaskState.submitted,
        }
        return status_mapping.get(custom_status, TaskState.working) 