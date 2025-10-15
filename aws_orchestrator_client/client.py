import asyncio
from typing import cast
import json
from uuid import uuid4

import asyncclick as click
import httpx
from pprint import pprint
from pprint import PrettyPrinter
# Add colorama for colored output
from colorama import init as colorama_init, Fore, Style
colorama_init(autoreset=True)

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    FilePart,
    FileWithBytes,
    GetTaskRequest,
    JSONRPCErrorResponse,
    Message,
    MessageSendConfiguration,
    MessageSendParams,
    Part,
    SendMessageRequest,
    SendStreamingMessageRequest,
    Task,
    TaskArtifactUpdateEvent,
    TaskQueryParams,
    TaskState,
    TaskStatusUpdateEvent,
    TextPart,
    Role,
)


def cprint(text, color=Fore.WHITE, end='\n', bold=False):
    style = Style.BRIGHT if bold else Style.NORMAL
    print(f"{style}{color}{text}{Style.RESET_ALL}", end=end)

pp = PrettyPrinter(width=80, compact=True)


@click.command()
@click.option('--agent', default='http://localhost:10102')
@click.option('--session', default=0)
@click.option('--history', default=False)
async def cli(
    agent,
    session,
    history,
):
    async with httpx.AsyncClient(timeout=30) as httpx_client:
        card_resolver = A2ACardResolver(httpx_client, agent)
        card = await card_resolver.get_agent_card()

        print('======= Agent Card ========')
        print(card.model_dump_json(exclude_none=True))

        client = A2AClient(httpx_client, agent_card=card)

        continue_loop = True
        streaming = card.capabilities.streaming
        # Generate context_id once per CLI run (session)
        import uuid
        context_id = session if session > 0 else uuid.uuid4().hex

        while continue_loop:
            print('=========  starting a new task ======== ')
            # Always start with taskId=None for a new user-initiated task
            taskId = None
            multi_turn = True
            while multi_turn:
                prompt = click.prompt(
                    f'{Fore.GREEN}\nWhat do you want to send to the agent? (:q or quit to exit){Style.RESET_ALL}'
                )
                if prompt == ':q' or prompt == 'quit':
                    continue_loop = False
                    break
                cprint(f'you: {prompt}', color=Fore.GREEN, bold=True)
                message = Message(
                    role=Role.user,
                    parts=[cast(Part, TextPart(text=prompt))],
                    messageId=str(uuid4()),
                    taskId=taskId,
                    contextId=context_id,
                )
                payload = MessageSendParams(
                    message=message,
                    configuration=MessageSendConfiguration(
                        acceptedOutputModes=['text'],
                    ),
                )
                taskResult = None
                message_obj = None
                if streaming:
                    response_stream = client.send_message_streaming(
                        SendStreamingMessageRequest(
                            id=str(uuid4()),
                            params=payload,
                        )
                    )
                    async for result in response_stream:
                        if isinstance(result.root, JSONRPCErrorResponse):
                            cprint('Error: ' + str(result.root.error), color=Fore.RED, bold=True)
                            multi_turn = False
                            break
                        event = result.root.result
                        context_id = event.contextId
                        if isinstance(event, Task):
                            taskId = event.id
                        elif isinstance(event, TaskStatusUpdateEvent) or isinstance(
                            event, TaskArtifactUpdateEvent
                        ):
                            taskId = event.taskId
                            if isinstance(event, TaskStatusUpdateEvent) and event.status.state == "completed":
                                cprint(f'✅ Task completed: {event.status.state}', color=Fore.GREEN, bold=True)
                                print(f'stream event => {event.model_dump_json(exclude_none=True)}')
                                multi_turn = False
                                break
                        elif isinstance(event, Message):
                            message_obj = event
                        cprint('bot_reply<server>:', color=Fore.CYAN, bold=True)
                        pp.pprint(event.model_dump(exclude_none=True))
                        cprint('  =========================', color=Fore.MAGENTA, bold=True)
                    # After streaming, fetch the final task result if available
                    if taskId:
                        try:
                            taskResult = await client.get_task(
                                GetTaskRequest(
                                    id=str(uuid4()),
                                    params=TaskQueryParams(id=taskId),
                                )
                            )
                            if not isinstance(taskResult.root, JSONRPCErrorResponse):
                                taskResult = taskResult.root.result
                        except Exception as e:
                            cprint(f'Failed to get task result: {e}', color=Fore.RED, bold=True)
                else:
                    try:
                        event = await client.send_message(
                            SendMessageRequest(
                                id=str(uuid4()),
                                params=payload,
                            )
                        )
                        if not isinstance(event.root, JSONRPCErrorResponse):
                            event = event.root.result
                        else:
                            cprint('Error: ' + str(event.root.error), color=Fore.RED, bold=True)
                            multi_turn = False
                            break
                    except Exception as e:
                        cprint('Failed to complete the call ' + str(e), color=Fore.RED, bold=True)
                        multi_turn = False
                        break
                    if not context_id:
                        context_id = event.contextId
                    if isinstance(event, Task):
                        if not taskId:
                            taskId = event.id
                        taskResult = event
                    elif isinstance(event, Message):
                        message_obj = event
                # Print message or task result
                if message_obj:
                    cprint('bot_reply<server>:', color=Fore.CYAN, bold=True)
                    pp.pprint(message_obj.model_dump(exclude_none=True))
                    cprint('  =========================', color=Fore.MAGENTA, bold=True)
                    multi_turn = False
                elif taskResult:
                    task_content = taskResult.model_dump_json(
                        exclude={
                            'history': {
                                '__all__': {
                                    'parts': {
                                        '__all__': {'file'},
                                    },
                                },
                            },
                        },
                        exclude_none=True,
                    )
                    try:
                        task_content_dict = json.loads(task_content)
                    except Exception:
                        task_content_dict = task_content
                    cprint('bot_reply<server>:', color=Fore.CYAN, bold=True)
                    pp.pprint(task_content_dict)
                    cprint('  =========================', color=Fore.MAGENTA, bold=True)
                    cprint(f'DEBUG: Received taskId: {getattr(taskResult, "id", None)}, contextId: {getattr(taskResult, "contextId", None)}', color=Fore.YELLOW, bold=True)
                    # Multi-turn: if input_required, prompt again with same taskId/contextId
                    if isinstance(taskResult, Task) and hasattr(taskResult, 'status') and hasattr(taskResult.status, 'state'):
                        state = TaskState(taskResult.status.state)
                    else:
                        state = TaskState.completed
                    if state.name == TaskState.input_required.name:
                        question = getattr(taskResult, 'question', None)
                        if question:
                            cprint(f'Agent asks: {question}', color=Fore.YELLOW, bold=True)
                        # Continue multi-turn loop with same taskId/context_id
                        continue
                    else:
                        # Task is complete, break multi-turn loop
                        multi_turn = False
                else:
                    # No valid result, break
                    multi_turn = False
            if history and continue_loop and taskId:
                print('========= history ======== ')
                task_response = await client.get_task(
                    GetTaskRequest(
                        id=str(uuid.uuid4()),
                        params=TaskQueryParams(id=taskId, historyLength=10)
                    )
                )
                print(
                    task_response.model_dump_json(
                        include={'result': {'history': True}}
                    )
                )


if __name__ == '__main__':
    asyncio.run(cli())