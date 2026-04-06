"""
A2A Executor for the AWS Orchestrator Agent.

Orchestrates the A2A protocol lifecycle:
  1. Extract query from incoming context (text or A2UI userAction)
  2. Resolve / create task
  3. Set up A2UI session (activation + catalog negotiation)
  4. Wrap query as ``Command(resume=...)`` when resuming an interrupt
  5. Dispatch A2UI schema event to client
  6. Stream agent responses → dispatch to response handlers

Three response types are handled:
  - **completed**       → artifact + final status
  - **input_required**  → HITL pause (``TaskState.input_required``)
  - **working**         → intermediate progress update
"""

import asyncio
import inspect
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, cast

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    DataPart,
    InvalidParamsError,
    Part,
    Message,
    Role,
    SendStreamingMessageSuccessResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)
from a2a.utils import new_agent_text_message, new_task, new_agent_parts_message
from a2a.utils.errors import ServerError
from a2ui.a2a import (
    A2UI_EXTENSION_BASE_URI,
    A2UI_MIME_TYPE,
    create_a2ui_part,
    get_a2ui_agent_extension,
    is_a2ui_part,
    parse_response_to_parts,
)
from langgraph.types import Command

from aws_orchestrator_agent.core.agents.types import AgentResponse, BaseAgent
from aws_orchestrator_agent.core.a2ui import (
    A2UI_CLIENT_CAPABILITIES_KEY,
    RenderContext,
    get_catalog_manager,
    get_registry,
)
from aws_orchestrator_agent.utils.logger import AgentLogger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

A2UI_EXTENSION_URI = f"{A2UI_EXTENSION_BASE_URI}/v0.8"

logger = AgentLogger("A2AExecutor")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_agent_message(parts: List[Part]) -> Message:
    """Create an agent message with a unique ``messageId``."""
    return Message(role=Role.agent, parts=parts, message_id=str(uuid.uuid4()))


# ---------------------------------------------------------------------------
# Stream Renderer — manages token‑stream formatting state
# ---------------------------------------------------------------------------

class _StreamRenderer:
    """Stateful helper for the token stream inside ``_stream_agent``.

    Responsibilities:
      • Maintains a **stable ``message_id``** (UUID) so the A2A frontend
        concatenates all ``final=False`` chunks into a single chat bubble.
      • Tracks whether a ``<details>`` thinking‑block is currently open
        and provides ``open_thinking()`` / ``close_thinking()`` to toggle it.
      • Provides a single ``emit(text)`` method that sends a ``TextPart``
        via the ``TaskUpdater``.

    All formatting constants live here, not scattered through if‑else branches.
    """

    _OPEN_TAG = "\n<details open>\n<summary><b>Show thinking</b></summary>\n\n"
    _CLOSE_TAG = "\n</details>\n\n"

    # Map internal node/source names to friendly display labels.
    # Names NOT in this map are Title-Cased automatically.
    # Names mapped to "" are suppressed (no label emitted).
    _LABEL_MAP: dict[str, str] = {
        "model": "",           # internal LangGraph node
        "tools": "",           # internal LangGraph node
        "": "",                # empty
        "coordinator": "Supervisor",
    }

    def __init__(self, updater: TaskUpdater, context_id: str, task_id: str) -> None:
        self._updater = updater
        self._ctx = context_id
        self._task = task_id
        self.message_id = str(uuid.uuid4())
        self._thinking_open = False
        self._current_agent: str = ""

    # ── public API ────────────────────────────────────────────────────

    async def emit_with_label(self, text: Any, meta: dict) -> None:
        """Emit text, injecting an agent label if the source changed.

        Large raw JSON blobs (tool output piped into the AI reasoning path)
        are suppressed — they are replaced with nothing so the user doesn't
        see hundreds of lines of raw data in the stream.
        """
        agent = self._resolve_agent(meta)
        if agent and agent != self._current_agent:
            self._current_agent = agent
            await self.emit(f"\n🤖 **{agent}**\n\n")

        content = str(text) if text else ""
        # Suppress raw JSON blobs leaking into the AI text stream.
        # These are tool results that Gemini re-narrates verbatim — they add
        # no value to the user and make the stream unreadable.
        stripped = content.strip()
        if len(stripped) > 300 and stripped.startswith("{") and stripped.endswith("}"):
            return  # silently drop — the tool badge already summarises it
        if len(stripped) > 300 and stripped.startswith("[") and stripped.endswith("]"):
            return  # same for JSON arrays

        await self.emit(content)

    async def emit(self, text: Any) -> None:
        """Send a text chunk to the client using the stable message ID."""
        content = str(text) if text else ""
        if not content:
            return
        msg = Message(
            role=Role.agent,
            parts=[Part(root=TextPart(text=content))],
            message_id=self.message_id,
            context_id=self._ctx,
            task_id=self._task,
        )
        await self._updater.update_status(TaskState.working, msg, final=False)
        await asyncio.sleep(0)  # yield to EventConsumer

    async def open_thinking(self) -> None:
        """Open a ``<details>`` thinking block if not already open."""
        if not self._thinking_open:
            await self.emit(self._OPEN_TAG)
            self._thinking_open = True

    async def close_thinking(self) -> None:
        """Close the ``<details>`` thinking block if currently open."""
        if self._thinking_open:
            await self.emit(self._CLOSE_TAG)
            self._thinking_open = False

    @classmethod
    def _resolve_agent(cls, meta: dict) -> str:
        """Return a display-ready agent label, or '' to suppress.

        Tries keys in priority order.  If a key maps to '' in _LABEL_MAP
        (i.e. it's internal plumbing), skip it and try the next key.
        Strips UUID suffixes like 'Requirements Analyser:697F0E55-...' so
        only the human-readable label is shown.
        """
        for key in ("agent_name", "node", "source"):
            raw = meta.get(key, "")
            if not raw:
                continue
            # Strip UUID suffix: 'Name:UUID' → 'Name'
            if ":" in raw:
                raw = raw.split(":")[0].strip()
            if raw in cls._LABEL_MAP:
                mapped = cls._LABEL_MAP[raw]
                if mapped:        # e.g. coordinator → Supervisor
                    return mapped
                continue          # e.g. model → "" → try next key
            # Auto-format: "tf_planner" → "Tf Planner"
            return raw.replace("_", " ").title()
        return ""


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class A2AExecutor(AgentExecutor):
    """A2A protocol executor for the AWS Orchestrator Agent.

    Responsibilities are split into small, testable methods:

    * **Query extraction** — ``_extract_query``, ``_extract_user_action``
    * **Task lifecycle**  — ``_resolve_task``, ``_setup_session``, ``_wrap_resume``
    * **Schema handshake** — ``_send_schema_event``
    * **Streaming loop**  — ``_stream_agent``
    * **Response dispatch** — ``_handle_completed``, ``_handle_input_required``,
      ``_handle_working``
    """

    # Class-level session store shared across instances for persistence
    _sessions: Dict[str, Dict[str, Any]] = {}

    def __init__(self, agent: BaseAgent) -> None:
        self.agent = agent
        self._catalog = get_catalog_manager()

    # ── public entry points ───────────────────────────────────────────

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute the full A2A request lifecycle."""
        logger.info(f"Executing agent {self.agent.name}")

        # 1. Extract query (text or A2UI userAction)
        query: Union[str, Command, None] = self._extract_query(context)

        # 2. Resolve or create task
        task = await self._resolve_task(context, event_queue)
        ctx_id = task.context_id

        # 3. Set up A2UI session
        session, use_ui = self._setup_session(context, ctx_id)

        # 4. Wrap as resume command if returning from interrupt
        query = self._wrap_resume(task, query)

        # 5. Dispatch schema event (deferred — schema is now sent just-in-time
        #    inside _handle_input_required / _handle_completed, right before the
        #    first beginRendering surface.  This prevents the client switching
        #    into A2UI-surface-only mode before streaming tokens have rendered.)
        updater = TaskUpdater(event_queue, task.id, ctx_id)

        # 6. Stream agent → dispatch response handlers
        await self._stream_agent(query, task, updater, event_queue, ctx_id, use_ui, session)

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Cancel an ongoing execution and clean up resources."""
        ctx_id = context.context_id or context.task_id or ""
        logger.info("Cancelling execution", extra={"context_id": ctx_id})

        self._sessions.pop(ctx_id, None)
        await self.agent.cleanup()

        task_id = context.task_id or ""
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=task_id,
                context_id=ctx_id,
                status=TaskStatus(
                    state=TaskState.canceled,
                    message=new_agent_text_message("Execution cancelled"),
                    timestamp=datetime.now().isoformat(),
                ),
                final=True,
            )
        )

    # ── Step 1: Query extraction ──────────────────────────────────────

    def _extract_query(self, context: RequestContext) -> Optional[str]:
        """Extract the user's query from the request context.

        Checks for plain text first, then falls back to parsing
        A2UI ``userAction`` DataParts.
        """
        query = context.get_user_input()
        if query:
            logger.debug(f"User query (text): {query[:120]}")
            return query

        # Try A2UI userAction
        if context.message and context.message.parts:
            action_query = self._extract_user_action(context.message.parts)
            if action_query:
                return action_query

        return query

    def _extract_user_action(self, parts: List[Part]) -> Optional[str]:
        """Parse A2UI ``userAction`` from message DataParts.

        Handles both standard A2UI parts (detected via SDK) and raw DataParts.
        For ``hitl_response`` actions, extracts the decision context.
        """
        for part in parts:
            user_action = self._get_user_action_from_part(part)
            if user_action is None:
                continue

            query = self._resolve_action_to_query(user_action)
            logger.info("Extracted A2UI userAction", extra={"action": query[:200]})
            return query

        return None

    @staticmethod
    def _get_user_action_from_part(part: Part) -> Optional[dict]:
        """Extract ``userAction`` dict from a Part, or None."""
        if is_a2ui_part(part):
            root = part.root
            if isinstance(root, DataPart) and isinstance(root.data, dict):
                return root.data.get("userAction")
        else:
            root = getattr(part, "root", part)
            if isinstance(root, DataPart) and isinstance(root.data, dict):
                return root.data.get("userAction")
        return None

    @staticmethod
    def _resolve_action_to_query(user_action: Any) -> str:
        """Convert a ``userAction`` dict into a query string for the agent.

        For ``hitl_response`` actions, extracts context items into a flat dict.

        When the context has **only** a ``decision`` key (simple approve/reject),
        returns the decision string directly — backward compatible with existing
        HITL flows (``request_human_input``, ``InterruptApprovalComponent``).

        When the context has **additional** keys (e.g., ``repository``, ``branch``
        from ``CommitApprovalComponent``), returns the full context as JSON so all
        fields reach the tool's ``interrupt()`` resume value.
        """
        if not isinstance(user_action, dict):
            return json.dumps(user_action)

        if user_action.get("name") != "hitl_response":
            return json.dumps(user_action)

        # Parse HITL context items into a flat dict
        ctx: Dict[str, Any] = {}
        for item in user_action.get("context", []):
            if not isinstance(item, dict):
                continue
            key = item.get("key")
            if not key:
                continue
            val = item.get("value")
            if isinstance(val, dict):
                # Resolve ValueRef shapes
                ctx[key] = (
                    val.get("literalString")
                    or val.get("valueString")
                    or val.get("literalNumber")
                    or val.get("literalBoolean")
                    or val.get("path")
                    or val
                )
            else:
                ctx[key] = val

        decision = ctx.get("decision", "").strip()

        # Parse formInputs (from A2UI / Google Chat standard)
        form_inputs = user_action.get("formInputs", {})
        if form_inputs:
            for k, v in form_inputs.items():
                # Google Chat format: {"stringInputs": {"value": ["..."]}}
                if isinstance(v, dict) and "stringInputs" in v:
                    strings = v["stringInputs"].get("value", [])
                    if strings:
                        ctx[k] = strings[0]
                # Simplified format: {"key": "..."}
                elif isinstance(v, str):
                    ctx[k] = v

        # If context has additional data beyond decision (e.g., repository,
        # branch for commit approval), return the full context so the tool
        # receives all the user-provided fields via interrupt() resume.
        non_decision_keys = {k for k in ctx if k != "decision" and ctx[k]}
        if decision and non_decision_keys:
            return json.dumps(ctx)

        # Simple decision-only context → return bare string (backward compat)
        if decision:
            return decision
        if ctx:
            return json.dumps(ctx)
        return json.dumps(user_action)

    # ── Step 2: Task resolution ───────────────────────────────────────

    @staticmethod
    async def _resolve_task(
        context: RequestContext,
        event_queue: EventQueue,
    ) -> Task:
        """Return the existing task or create a new one."""
        task = context.current_task
        if task:
            return task

        if context.message is None:
            raise ServerError(error=InvalidParamsError())

        task = new_task(context.message)
        await event_queue.enqueue_event(task)
        return task

    # ── Step 3: Session setup ─────────────────────────────────────────

    def _setup_session(
        self,
        context: RequestContext,
        context_id: str,
    ) -> tuple[Dict[str, Any], bool]:
        """Activate A2UI and negotiate catalog schema.

        Returns:
            ``(session_dict, use_ui)``
        """
        use_ui = self._try_activate_a2ui(context)
        session = self._get_or_create_session(context_id)
        session["a2ui_enabled"] = use_ui

        # Negotiate schema once per session
        if use_ui and not session.get("a2ui_schema"):
            client_caps = self._get_client_capabilities(context)
            catalog_uri, merged_schema = self._catalog.select_catalog(client_caps)
            session["a2ui_catalog_uri"] = catalog_uri
            session["a2ui_schema"] = merged_schema

        return session, use_ui

    def _try_activate_a2ui(self, context: RequestContext) -> bool:
        """Check whether A2UI should be activated for this request."""
        message = context.message
        if message and hasattr(message, "metadata") and message.metadata:
            client_caps = message.metadata.get(A2UI_CLIENT_CAPABILITIES_KEY, {})
            if client_caps:
                logger.info("A2UI client capabilities detected", extra={"capabilities": client_caps})
                return True

        if hasattr(context, "extensions") and context.extensions:  # type: ignore[attr-defined]
            if A2UI_EXTENSION_URI in context.extensions:  # type: ignore[attr-defined]
                return True

        # Default: enable A2UI for this agent
        return True

    def _get_or_create_session(self, context_id: str) -> Dict[str, Any]:
        """Return the session dict for a context, creating if needed."""
        if context_id not in self._sessions:
            logger.info("Creating new session", extra={"context_id": context_id})
            self._sessions[context_id] = {
                "a2ui_enabled": False,
                "a2ui_catalog_uri": None,
                "a2ui_schema": None,
                "pending_interrupts": {},
                "task_state": TaskState.submitted,
            }
        return self._sessions[context_id]

    @staticmethod
    def _get_client_capabilities(context: RequestContext) -> Optional[Dict[str, Any]]:
        """Extract A2UI client capabilities from the request metadata."""
        msg = context.message
        if msg and hasattr(msg, "metadata") and msg.metadata:
            return msg.metadata.get(A2UI_CLIENT_CAPABILITIES_KEY, {})
        return None

    # ── Step 4: Resume wrapping ───────────────────────────────────────

    @staticmethod
    def _wrap_resume(task: Task, query: Any) -> Union[str, Command, None]:
        """Wrap ``query`` in ``Command(resume=...)`` if the task is paused."""
        if not (task and hasattr(task, "status") and hasattr(task.status, "state")):
            return query

        if task.status.state == TaskState.input_required:
            logger.info(
                "Resuming from input_required — wrapping as Command(resume=...)",
                extra={"task_id": task.id, "preview": str(query)[:100] if query else "None"},
            )
            return Command(resume=query)

        return query

    # ── Step 5: Schema event ──────────────────────────────────────────

    @staticmethod
    async def _send_schema_event(
        updater: TaskUpdater,
        session: Dict[str, Any],
        task: Task,
    ) -> None:
        """Send the A2UI schema state-delta just before the first surface render.

        Called from ``_handle_input_required`` and ``_handle_completed`` right
        before they emit a ``beginRendering`` surface, so the client receives
        the catalog schema at the moment it actually needs it — NOT before
        streaming starts (which would switch the client into surface-only mode
        and suppress TextPart token updates).

        Idempotent: marks ``a2ui_schema_sent`` in session so it fires only once
        per session, even if multiple surfaces are rendered.
        """
        if session.get("a2ui_schema_sent") or not session.get("a2ui_schema"):
            return

        schema_payload = {
            "actions": {
                "state_delta": {
                    "system:a2ui_enabled": True,
                    "system:a2ui_schema": session["a2ui_schema"],
                    "system:a2ui_catalog_uri": session["a2ui_catalog_uri"],
                }
            }
        }
        logger.info("Dispatching A2UI schema event (just-in-time)", extra={"task_id": task.id})
        await updater.update_status(
            TaskState.working,
            _new_agent_message([create_a2ui_part(schema_payload)]),
            final=False,
        )
        session["a2ui_schema_sent"] = True
        await asyncio.sleep(0)  # yield for EventConsumer

    # ── Step 6: Agent streaming ───────────────────────────────────────

    def _make_stream_message_text(
        self, text: str, message_id: Optional[str], context_id: str, task_id: str
    ) -> Message:
        """Create a plain text streaming message with a stable ID."""
        if not message_id:
            return new_agent_text_message(text, context_id, task_id)
        return Message(
            role=Role.agent,
            parts=[Part(root=TextPart(text=text))],
            message_id=message_id,
            context_id=context_id,
            task_id=task_id,
        )

    def _make_stream_message_parts(
        self, parts: List[Part], message_id: Optional[str], context_id: str, task_id: str
    ) -> Message:
        """Create a parts-based message with a stable ID."""
        if not message_id:
            return new_agent_parts_message(parts, context_id, task_id)
        return Message(
            role=Role.agent,
            parts=parts,
            message_id=message_id,
            context_id=context_id,
            task_id=task_id,
        )



    async def _stream_agent(
        self,
        query: Union[str, Command, None],
        task: Task,
        updater: TaskUpdater,
        event_queue: EventQueue,
        context_id: str,
        use_ui: bool,
        session: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Stream agent responses and dispatch to the appropriate handler.

        Uses a ``_StreamRenderer`` helper to manage thinking-block state
        and agent-transition labels, keeping this method focused on the
        dispatch logic only.
        """
        logger.info(
            "Starting agent stream",
            extra={
                "task_id": task.id,
                "context_id": context_id,
                "is_resume": isinstance(query, Command),
            },
        )

        try:
            stream_query: Union[str, Command] = (
                query if isinstance(query, Command) else str(query or "")
            )
            agent_stream = self.agent.stream(  # type: ignore[arg-type]
                stream_query,
                context_id,
                task.id,
                use_ui=use_ui,
            )
            if not inspect.isasyncgen(agent_stream):
                agent_stream = await agent_stream  # type: ignore[misc]

            renderer = _StreamRenderer(updater, context_id, task.id)

            async for item in agent_stream:  # type: ignore[union-attr]
                # ── Forward A2A inter-agent events directly ───────────
                root = getattr(item, "root", None)
                if root is not None and isinstance(root, SendStreamingMessageSuccessResponse):
                    event = root.result
                    if isinstance(event, (TaskStatusUpdateEvent, TaskArtifactUpdateEvent)):
                        await event_queue.enqueue_event(event)
                    continue

                # ── Classify the item ─────────────────────────────────
                is_complete = getattr(item, "is_task_complete", False)
                needs_input = getattr(item, "require_user_input", False)
                meta = item.metadata or {}
                is_tool = (
                    item.response_type == "token"
                    and meta.get("message_type") in ("tool_call", "tool_result")
                )

                # ── Terminal events ───────────────────────────────────
                if is_complete:
                    await renderer.close_thinking()
                    await self._handle_completed(item, updater, task, context_id, use_ui, renderer.message_id, session)
                    return
                if needs_input:
                    await renderer.close_thinking()
                    await self._handle_input_required(item, updater, task, context_id, use_ui, renderer.message_id, session)
                    break

                # ── Tool progress → inside thinking block ─────────────
                if is_tool:
                    await renderer.open_thinking()
                    await renderer.emit_with_label(item.content, meta)
                    continue

                # ── AI text token → outside thinking block ────────────
                if item.response_type == "token":
                    await renderer.close_thinking()
                    await renderer.emit_with_label(item.content, meta)
                    continue

                # ── Structured / other working updates ────────────────
                await renderer.close_thinking()
                await self._handle_working(item, updater, task, context_id, use_ui, renderer.message_id)

        except Exception as e:
            logger.error(f"Exception in agent stream: {e}")
            raise

    # ── Response handlers ─────────────────────────────────────────────

    async def _handle_completed(
        self,
        item: AgentResponse,
        updater: TaskUpdater,
        task: Task,
        context_id: str,
        use_ui: bool,
        stream_message_id: str = "",
        session: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Handle a **completed** response from the agent."""
        logger.info("Task marked complete by agent", extra={"task_id": task.id})

        if use_ui:
            if item.response_type == "token":
                # Close the stream directly with an empty final chunk (no A2UI card)
                await updater.update_status(
                    TaskState.completed,
                    self._make_stream_message_text("", stream_message_id, context_id, task.id),
                    final=True,
                )
            else:
                # Send schema just-in-time before the first beginRendering surface
                if session:
                    await self._send_schema_event(updater, session, task)

                content = item.content or "Task completed successfully."
                if isinstance(content, str) and not content.strip():
                    content = "Task completed successfully."

                parts = self._build_a2ui_parts(
                    content=content,
                    status="completed",
                    is_task_complete=True,
                    response_type=item.response_type,
                    metadata=item.metadata,
                )
                await updater.update_status(
                    TaskState.completed,
                    self._make_stream_message_parts(parts, stream_message_id, context_id, task.id),
                    final=True,
                )
        else:
            # Plain text / data fallback
            if item.response_type == "data":
                part: Part = cast(Part, DataPart(data=item.content))
            else:
                part = cast(Part, TextPart(text=self._content_to_str(item.content)))
            await updater.add_artifact([part], name=f"{self.agent.name}-result")
            await updater.update_status(
                TaskState.completed,
                self._make_stream_message_text("Task completed successfully.", stream_message_id, context_id, task.id),
                final=True,
            )

        await self._safe_complete(updater, task)

    async def _handle_input_required(
        self,
        item: AgentResponse,
        updater: TaskUpdater,
        task: Task,
        context_id: str,
        use_ui: bool,
        stream_message_id: str = "",
        session: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Handle an **input_required** response (HITL interrupt)."""
        logger.info("Agent requires user input", extra={"task_id": task.id})

        if use_ui:
            if item.response_type == "token":
                # Just close the stream natively
                await updater.update_status(
                    TaskState.input_required,
                    self._make_stream_message_text("", stream_message_id, context_id, task.id),
                    final=True,
                )
            else:
                # Send schema just-in-time before the first beginRendering surface
                if session:
                    await self._send_schema_event(updater, session, task)

                parts = self._build_a2ui_parts(
                    content=item.content,
                    status="input_required",
                    is_task_complete=False,
                    require_user_input=True,
                    response_type=item.response_type,
                    metadata=item.metadata,
                )
                await updater.update_status(
                    TaskState.input_required,
                    self._make_stream_message_parts(parts, stream_message_id, context_id, task.id),
                    final=True,
                )
        else:
            text = "Please provide input." if item.response_type == "token" else self._content_to_str(item.content)
            await updater.update_status(
                TaskState.input_required,
                self._make_stream_message_text(text, stream_message_id, context_id, task.id),
                final=True,
            )

    async def _handle_working(
        self,
        item: AgentResponse,
        updater: TaskUpdater,
        task: Task,
        context_id: str,
        use_ui: bool,
        stream_message_id: Optional[str] = None,
    ) -> None:
        """Handle an intermediate **working** update.

        Two routing tiers (both follow the ci-executor pattern of
        ``add_artifact`` for anything the client must actually render):

        **Token stream** (``response_type="token"``):
            Sent as a lightweight ``TextPart`` via ``update_status`` only —
            no artifact.  Uses a **stable** ``stream_message_id`` so the
            A2A frontend can concatenate all incremental chunks within the
            same conversational turn.

        **All other working updates** (``tool_call``, ``tool_result``,
        ``ai_text``, ``text``, etc.):
            Go through ``add_artifact`` + ``update_status`` (parts path)
            when ``use_ui`` is active.  This is the only reliable path —
            many A2A clients silently ignore ``TaskStatusUpdateEvent`` that
            carry only a ``TextPart`` with ``final=False``.

        When ``use_ui=False`` all updates use the plain text path.
        """
        if use_ui and item.response_type != "token":
            # Structured/activity updates → full A2UI artifact
            parts = self._build_a2ui_parts(
                content=item.content,
                status="working",
                is_task_complete=False,
                response_type=item.response_type,
                metadata=item.metadata,
            )
            await updater.add_artifact(parts, name=f"{self.agent.name}-intermediate")
            await updater.update_status(
                TaskState.working,
                new_agent_parts_message(parts, context_id, task.id),
                final=False,
            )
        else:
            # Token stream or no-UI fallback → lightweight plain text.
            # Use the stable stream_message_id so the frontend concatenates
            # all chunks instead of treating each as a separate message.
            text = self._content_to_str(item.content)
            msg = self._make_stream_message_text(text, stream_message_id, context_id, task.id)
            await updater.update_status(
                TaskState.working,
                msg,
                final=False,
            )
        await asyncio.sleep(0)  # yield control to EventConsumer (mirrors ci-executor)

    # ── Shared utilities ──────────────────────────────────────────────

    def _build_a2ui_parts(
        self,
        content: Any,
        status: str = "working",
        is_task_complete: bool = False,
        require_user_input: bool = False,
        response_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Part]:
        """Build A2UI Parts via the Component Registry."""
        meta = metadata or {}
        meta.setdefault("status", status)

        ctx = RenderContext(
            content=content,
            status=meta.get("status", "working"),
            is_task_complete=is_task_complete,
            require_user_input=require_user_input,
            response_type=response_type,
            metadata=meta,
        )
        return get_registry().build_parts(ctx)

    @staticmethod
    def _content_to_str(content: Any) -> str:
        """Convert content to a display-friendly string."""
        if isinstance(content, dict):
            return (
                content.get("summary")
                or content.get("question")
                or content.get("message")
                or json.dumps(content, indent=2)
            )
        return str(content) if content else "Processing..."

    @staticmethod
    def _map_status(custom_status: str) -> TaskState:
        """Map custom status strings to ``TaskState`` enum values."""
        return {
            "working": TaskState.working,
            "input_required": TaskState.input_required,
            "completed": TaskState.completed,
            "failed": TaskState.failed,
            "error": TaskState.failed,
            "submitted": TaskState.submitted,
        }.get(custom_status, TaskState.working)

    @staticmethod
    async def _safe_complete(updater: TaskUpdater, task: Task) -> None:
        """Call ``updater.complete()`` with graceful handling of terminal-state errors."""
        try:
            await updater.complete()
            logger.info("Task completed", extra={"task_id": task.id})
        except RuntimeError as e:
            if "already in a terminal state" in str(e):
                logger.info("Task already terminal, skipping complete()", extra={"task_id": task.id})
            else:
                raise

    def get_supported_catalogs(self) -> List[str]:
        """Return all catalog URIs this executor supports."""
        return self._catalog.get_supported_catalog_ids()