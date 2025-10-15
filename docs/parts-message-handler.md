# PartsMessageHandler: Structured Parts-Based Message Handling

## Overview

`PartsMessageHandler` is a utility class designed to centralize the construction, parsing, and validation of A2A protocol Parts-based messages within the AWS Orchestrator Agent framework. It provides a robust, extensible, and testable foundation for agent-to-agent communication using structured message parts (e.g., `TextPart`, `DataPart`).

## Key Features
- **Centralized Construction:** Create Part objects from type and content in a consistent, validated manner.
- **Parsing:** Convert Part objects into serializable dictionaries for logging, debugging, or downstream processing.
- **Validation:** Ensure that Parts conform to protocol requirements before sending or storing.
- **Extensibility:** Easily add support for new Part types (e.g., `FilePart`, `ImagePart`).

## Usage

### Construction
```python
from aws_orchestrator_agent.core.parts_handler import PartsMessageHandler

# Construct a TextPart
text_part = PartsMessageHandler.construct_part("TextPart", "hello world")

# Construct a DataPart
data_part = PartsMessageHandler.construct_part("DataPart", {"foo": "bar"})
```

### Parsing
```python
parsed = PartsMessageHandler.parse_part(text_part)
# Output: {"type": "TextPart", "content": "hello world"}
```

### Validation
```python
is_valid = PartsMessageHandler.validate_part(data_part)
# Returns True if valid, False otherwise
```

## Integration Example

In the agent executor, use the handler to construct and validate Parts before adding them as artifacts:

```python
artifact_part = PartsMessageHandler.construct_part(
    "DataPart" if item.response_type == 'data' else "TextPart",
    item.content
)
if not PartsMessageHandler.validate_part(artifact_part):
    raise ValueError("Invalid Part constructed for task artifact")
await updater.add_artifact([artifact_part], name=f'{self.agent.name}-result')
```

## Extending for New Part Types

To add support for a new Part type (e.g., `FilePart`):
1. Import the new Part class from the A2A SDK.
2. Add it to the `SUPPORTED_PART_TYPES` dictionary in `PartsMessageHandler`.
3. Extend the `construct_part`, `parse_part`, and `validate_part` methods to handle the new type.

Example:
```python
from a2a.types import FilePart

# In SUPPORTED_PART_TYPES:
"FilePart": FilePart,

# In construct_part:
elif part_type == "FilePart":
    return FilePart(file=content)

# In parse_part and validate_part: add corresponding logic
```

## Testing

- **Unit Tests:** Ensure construction, parsing, and validation work for all supported Part types, including error cases.
- **Integration Tests:** Verify that Parts are correctly created and added as artifacts during agent execution, and that invalid Parts are rejected.

See `tests/test_a2a_executor.py` and `tests/test_a2a_executor_integration.py` for examples.

## Best Practices
- Always validate Parts before sending or storing them.
- Use the handler for all Part construction to ensure protocol compliance and future extensibility.
- Document and test any new Part types added to the system.

---

For further details, see the implementation in `core/parts_handler.py` and usage in `core/a2a_executor.py`.

---

# Task State Lifecycle Integration: A2A SDK vs. Internal TaskLifecycleManager

## Enum Usage and Mapping

When integrating the A2A SDK with the internal TaskLifecycleManager, it is critical to use the correct `TaskState` enum for each context:

| Context                | Enum to Use                                      | Example Value         |
|------------------------|--------------------------------------------------|----------------------|
| A2A SDK Task           | `a2a.types.TaskState` (import as A2ATaskState)   | `A2ATaskState.submitted` |
| Internal Lifecycle     | `aws_orchestrator_agent.schemas.task_lifecycle.TaskState` | `TaskState.SUBMITTED`    |

- **A2A SDK TaskState** uses lowercase values (e.g., `submitted`, `working`).
- **Internal TaskLifecycleManager TaskState** uses uppercase values (e.g., `SUBMITTED`, `WORKING`).

## Example: Correct Enum Usage in Tests and Production Code

```python
# For A2A SDK Task creation (e.g., in tests):
from a2a.types import Task, TaskStatus, TaskState as A2ATaskState

task = Task(
    id="t1",
    contextId="c1",
    status=TaskStatus(state=A2ATaskState.submitted)
)

# For all lifecycle assertions and transitions (internal):
from aws_orchestrator_agent.schemas.task_lifecycle import TaskState
assert task_history.transitions[0].to_state == TaskState.SUBMITTED
```

## Mapping in the Executor

The executor is responsible for mapping between A2A SDK states and internal states:
- When receiving a state from the A2A SDK, map it to the internal uppercase value for lifecycle management.
- When sending a state to the A2A SDK, map from the internal uppercase value to the SDK’s lowercase value.

## Best Practices for Integration
- **Always use the correct enum for the context:**
  - Use `A2ATaskState` for A2A SDK objects and communication.
  - Use internal `TaskState` for all lifecycle management, assertions, and transitions.
- **In tests:**
  - Use `A2ATaskState` for test setup involving A2A SDK objects.
  - Use internal `TaskState` for all assertions about task history and state transitions.
- **In the executor:**
  - Use mapping functions to convert between enums as needed.

## Why This Matters
- Using the correct enum ensures compatibility with both the A2A protocol and your internal lifecycle logic.
- It prevents subtle bugs and test failures due to enum mismatches.
- It keeps your codebase maintainable and clear for future contributors.

---

For more details, see the mapping logic in `core/a2a_executor.py` and the lifecycle management in `core/task_lifecycle.py`. 