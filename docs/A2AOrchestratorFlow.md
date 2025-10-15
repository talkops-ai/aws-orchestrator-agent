# A2A Protocol Integration

This module provides the core integration between AWS Orchestrator Agent and the A2A (Agent-to-Agent) protocol using the `a2a-sdk` library.

## Overview

The A2A protocol integration consists of several key components:

- **AWSOrchestratorAgentExecutor**: Bridges our BaseAgent implementations with the A2A SDK
- **BaseAgent Interface**: Abstract base class that all agents must implement
- **A2AIntegrationManager**: High-level manager for setting up multiple agents
- **Example Agents**: Sample implementations demonstrating the integration

## Key Components

### AWSOrchestratorAgentExecutor

The main executor class that integrates our agents with the A2A protocol:

```python
from aws_orchestrator_agent.core import AWSOrchestratorAgentExecutor
from aws_orchestrator_agent.types import BaseAgent

# Create an executor for your agent
executor = AWSOrchestratorAgentExecutor(your_agent)
```

The executor handles:
- Request validation
- Task creation and management
- Agent execution and streaming
- Status updates and artifact management
- Error handling and cleanup

### BaseAgent Interface

All agents must implement the `BaseAgent` interface:

```python
from aws_orchestrator_agent.types import BaseAgent, AgentResponse
from typing import AsyncGenerator

class YourAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "your-agent-name"
    
    async def stream(
        self, 
        query: str, 
        context_id: str, 
        task_id: str
    ) -> AsyncGenerator[AgentResponse, None]:
        # Your agent logic here
        yield AgentResponse(
            content="Processing...",
            metadata={"status": "working"}
        )
        
        # Final response
        yield AgentResponse(
            content="Task completed",
            is_task_complete=True,
            metadata={"status": "completed"}
        )
```

### A2AIntegrationManager

High-level manager for setting up multiple agents:

```python
from aws_orchestrator_agent.core import A2AIntegrationManager
from aws_orchestrator_agent.types import AgentConfig

# Create the manager
manager = A2AIntegrationManager()
await manager.initialize()

# Register agents
config = AgentConfig(name="my-agent", agent_type="custom")
handler = manager.register_agent(your_agent, config)

# Get the request handler for A2A protocol
request_handler = manager.get_request_handler("my-agent")
```

## Integration with DefaultRequestHandler

The executor works seamlessly with the A2A SDK's `DefaultRequestHandler`:

```python
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.server.tasks import TaskStore
from a2a.server.events import InMemoryQueueManager

# Create A2A components
task_store = TaskStore()
queue_manager = InMemoryQueueManager()

# Create the request handler
request_handler = DefaultRequestHandler(
    agent_executor=executor,
    task_store=task_store,
    queue_manager=queue_manager,
    push_notifier=None,  # Optional
)
```

## Example Usage

### Simple Single Agent

```python
from aws_orchestrator_agent.core import create_single_agent_handler
from aws_orchestrator_agent.types import AgentConfig

# Create your agent
config = AgentConfig(name="echo-agent", agent_type="echo")
agent = EchoAgent(config)

# Create the request handler
handler = create_single_agent_handler(agent, config)
```

### Multiple Agents

```python
from aws_orchestrator_agent.core import A2AIntegrationManager

# Create the manager
manager = A2AIntegrationManager()
await manager.initialize()

# Register multiple agents
agents = create_example_agents()
for agent_name, (agent, config) in agents.items():
    handler = manager.register_agent(agent, config)

# Get handlers for specific agents
echo_handler = manager.get_request_handler("echo-agent")
example_handler = manager.get_request_handler("example-agent")
```

## Status Mapping

The executor automatically maps custom status strings to A2A TaskState values:

- `'working'` → `TaskState.working`
- `'completed'` → `TaskState.completed`
- `'failed'` → `TaskState.failed`
- `'input_required'` → `TaskState.input_required`
- `'cancelled'` → `TaskState.cancelled`

## Error Handling

The executor provides comprehensive error handling:

- **Validation Errors**: Invalid requests are rejected with appropriate error messages
- **Execution Errors**: Agent execution errors are caught and task status is updated to failed
- **Stream Errors**: Streaming errors are handled gracefully with proper cleanup

## Testing

Run the tests to verify the implementation:

```bash
pytest tests/test_a2a_executor.py -v
```

## Best Practices

1. **Agent Implementation**:
   - Always implement the `BaseAgent` interface
   - Use proper status mapping in metadata
   - Handle errors gracefully in the stream method
   - Implement proper cleanup in the cleanup method

2. **Executor Usage**:
   - Validate requests before processing
   - Use appropriate logging levels
   - Handle task lifecycle properly
   - Implement proper error handling

3. **Integration**:
   - Use the `A2AIntegrationManager` for multiple agents
   - Configure proper timeouts and limits
   - Implement proper cleanup procedures
   - Use appropriate task store implementations for production

## Architecture

```
┌─────────────────┐    ┌─────────────────────┐    ┌─────────────────┐
│   BaseAgent     │    │ AWSOrchestrator     │    │ DefaultRequest  │
│   Interface     │───▶│ AgentExecutor       │───▶│ Handler         │
│                 │    │                     │    │                 │
└─────────────────┘    └─────────────────────┘    └─────────────────┘
         │                       │                           │
         │                       │                           │
         ▼                       ▼                           ▼
┌─────────────────┐    ┌─────────────────────┐    ┌─────────────────┐
│   AgentResponse │    │   TaskUpdater       │    │   TaskStore     │
│   Objects       │    │   EventQueue        │    │   QueueManager  │
└─────────────────┘    └─────────────────────┘    └─────────────────┘
```

This architecture ensures clean separation of concerns and proper integration with the A2A protocol. 