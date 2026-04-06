# Supervisor Agent → Generator Swarm Flow Documentation

## 📋 Overview

This document explains the complete flow from the **Supervisor Agent** (main client-facing agent) to the **Generator Swarm** (Planning Stage), including the data handoff structure and state management.

## 🔄 Complete Flow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CLIENT REQUEST                           │
│  "Create a VPC module with public/private subnets"         │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                SUPERVISOR AGENT                             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  • Main client-facing agent                            │ │
│  │  • Orchestrates all subgraph agents                    │ │
│  │  • Manages workflow state and routing                  │ │
│  │  • Handles human-in-the-loop interruptions             │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│            PLANNER SUB-SUPERVISOR                           │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  • Analyzes requirements and creates execution plans    │ │
│  │  • Uses specialized sub-agents (requirements, planner) │ │
│  │  │  • Requirements Analyzer Agent                      │ │
│  │  │  • Execution Planner Agent                          │ │
│  │  • Outputs comprehensive execution plan                │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│              GENERATOR SWARM                                │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  • Planning Stage (Current Focus)                      │ │
│  │  • Receives execution plan from planner                │ │
│  │  • Coordinates 4 specialized agents:                   │ │
│  │  │  • Resource Configuration Agent                     │ │
│  │  │  • Variable Definition Agent                        │ │
│  │  │  • Data Source Agent                                │ │
│  │  │  • Local Values Agent                               │ │
│  │  • Generates complete Terraform module                 │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 🎯 Key Components Analysis

### 1. **Supervisor Agent** (`supervisor_agent.py`)

#### **Role**: Main client-facing orchestrator
- **Entry Point**: Receives client requests
- **Orchestration**: Manages all subgraph agents using `langgraph-supervisor`
- **State Management**: Maintains `SupervisorState` with agent-specific data
- **Handoff Management**: Uses custom handoff tools for agent coordination

#### **Key Methods**:
```python
class CustomSupervisorAgent(BaseAgent):
    def __init__(self, agents: List[BaseSubgraphAgent], ...):
        # Initialize with centralized config and LLM
        # Build supervisor graph using langgraph-supervisor
        # Create handoff tools for all agents
    
    async def stream(self, query_or_command, context_id, task_id):
        # Main streaming interface for client requests
        # Handles both initial requests and resume commands
        # Manages human-in-the-loop interruptions
    
    def _merge_agent_state_back_to_supervisor(self, agent_name, agent_state, supervisor_state):
        # Merges agent results back into supervisor state
        # Handles planner_data → generation_data flow
```

#### **State Structure**:
```python
class SupervisorState(BaseModel):
    # Core workflow fields
    user_request: str
    session_id: Optional[str]
    task_id: Optional[str]
    status: WorkflowStatus
    current_agent: Optional[AgentType]
    
    # Agent-specific data storage
    planner_data: Optional[Dict[str, Any]] = None      # ← From Planner Sub-Supervisor
    generation_data: Optional[Dict[str, Any]] = None   # ← To Generator Swarm
    validation_data: Optional[Dict[str, Any]] = None
    editor_data: Optional[Dict[str, Any]] = None
    security_data: Optional[Dict[str, Any]] = None
    cost_data: Optional[Dict[str, Any]] = None
    
    # Human-in-the-loop
    human_approval_required: bool = False
    approval_context: Optional[Dict[str, Any]] = None
    question: Optional[str] = None
```

### 2. **Planner Sub-Supervisor** (`planner_sub_supervisor.py`)

#### **Role**: Strategic planning and execution plan generation
- **Input**: User request from supervisor
- **Processing**: Uses specialized planning agents
- **Output**: Comprehensive execution plan (like `execution_planner_5.json`)

#### **Key Flow**:
```python
class PlannerSubSupervisorAgent(BaseSubgraphAgent):
    def input_transform(self, supervisor_state: SupervisorState) -> PlannerSupervisorState:
        # Transform supervisor state to planner state
        # Extract user_request, session_id, task_id
        # Initialize planning workflow state
    
    def output_transform(self, planner_state: PlannerSupervisorState) -> Dict[str, Any]:
        # Transform planner results back to supervisor format
        return {
            "agent_result": {
                "execution_plan": planner_state.planning_results.execution_plan,
                "requirements_analysis": planner_state.requirements_data,
                "security_considerations": planner_state.security_n_best_practices_evaluator_data,
                "module_structure": planner_state.execution_data.module_structure_plan
            },
            "agent_status": "completed" if planner_state.workflow_state.planning_complete else "in_progress"
        }
```

#### **Output Structure** (matches `execution_planner_5.json`):
```json
{
  "service_name": "Amazon VPC",
  "module_name": "terraform-aws-vpc-advanced",
  "target_environment": "prod",
  "terraform_files": [...],
  "variable_definitions": [...],
  "local_values": [...],
  "data_sources": [...],
  "resource_configurations": [...],
  "outputs": [...],
  "dependencies": [...],
  "security_considerations": [...],
  "cost_estimates": [...],
  "deployment_strategy": {...}
}
```

### 3. **Generator Swarm** (`generator_swarm.py`)

#### **Role**: Tactical implementation of execution plan
- **Input**: Execution plan from planner (via supervisor)
- **Processing**: Coordinates 4 specialized agents
- **Output**: Complete Terraform module files

#### **State Structure**:
```python
class GeneratorStageState(TypedDict):
    # Core swarm fields
    messages: Annotated[List[BaseMessage], add_messages]
    active_agent: str
    
    # Planning stage management
    stage_status: str = "planning_active"
    planning_progress: Dict[str, float] = {}
    
    # Agent coordination
    agent_status_matrix: Dict[str, GeneratorAgentStatus] = {
        "resource_configuration_agent": GeneratorAgentStatus.INACTIVE,
        "variable_definition_agent": GeneratorAgentStatus.INACTIVE,
        "data_source_agent": GeneratorAgentStatus.INACTIVE,
        "local_values_agent": GeneratorAgentStatus.INACTIVE
    }
    
    # Agent workspaces with planner data
    agent_workspaces: Dict[str, Dict[str, Any]] = {
        "resource_configuration_agent": {
            "generated_resources": [],
            "planner_input": planner_output["resource_configurations"]  # ← Planner data
        },
        "variable_definition_agent": {
            "generated_variables": [],
            "planner_input": planner_output["variable_definitions"]  # ← Planner data
        },
        # ... etc
    }
```

## 🔄 Detailed Handoff Flow

### **Step 1: Client Request to Supervisor**

```python
# Client sends request
query = "Create a VPC module with public/private subnets"

# Supervisor receives and initializes
supervisor_state = SupervisorState(
    user_request=query,
    session_id=context_id,
    task_id=task_id,
    status=WorkflowStatus.PENDING,
    current_agent=AgentType.PLANNER
)
```

### **Step 2: Supervisor → Planner Sub-Supervisor**

```python
# Supervisor uses handoff tool
handoff_to_planner_sub_supervisor(
    task_description="Analyze requirements and create execution plan for VPC module",
    state=supervisor_state
)

# Handoff tool creates Command
return Command(
    goto="planner_sub_supervisor",
    graph=Command.PARENT,
    update={
        "messages": messages + [tool_message],
        "active_agent": "planner_sub_supervisor",
        "task_description": task_description,
        "user_request": user_request,
        "session_id": session_id,
        "task_id": task_id,
        "status": "in_progress"
    }
)
```

### **Step 3: Planner Sub-Supervisor Processing**

```python
# Planner processes request using specialized agents
planner_state = PlannerSupervisorState(
    user_request=user_request,
    session_id=session_id,
    task_id=task_id,
    workflow_state=PlanningWorkflowState(
        current_phase="requirements_analysis"
    )
)

# Planner coordinates:
# 1. Requirements Analyzer Agent
# 2. Execution Planner Agent
# 3. Generates comprehensive execution plan
```

### **Step 4: Planner → Supervisor (Return)**

```python
# Planner returns results via output_transform
planner_results = {
    "agent_result": {
        "execution_plan": {
            "service_name": "Amazon VPC",
            "module_name": "terraform-aws-vpc-advanced",
            "resource_configurations": [...],
            "variable_definitions": [...],
            "data_sources": [...],
            "local_values": [...],
            "dependencies": [...],
            "security_considerations": [...],
            "cost_estimates": [...]
        }
    },
    "agent_status": "completed"
}

# Supervisor merges results
supervisor_updates = {
    "messages": agent_state.get("messages", []),
    "planner_data": agent_state.get("agent_result", {}),  # ← Key mapping
    "status": WorkflowStatus.COMPLETED,
    "current_agent": None  # Planning complete, move to next phase
}
```

### **Step 5: Supervisor → Generator Swarm**

```python
# Supervisor determines next agent (generation_agent)
# Uses StateTransformer to convert supervisor state to generation state

generation_state = StateTransformer.supervisor_to_generation(supervisor_state)
# This extracts planner_data and transforms it for generation

# Supervisor hands off to generator swarm
handoff_to_generation_agent(
    task_description="Generate Terraform module from execution plan",
    state=supervisor_state
)
```

### **Step 6: Generator Swarm Processing**

```python
# Generator Swarm receives execution plan via supervisor state
# Transforms planner data into GeneratorStageState

def transform_planner_input_to_state(planner_output: Dict[str, Any]) -> GeneratorStageState:
    return GeneratorStageState(
        # Core swarm fields
        messages=[HumanMessage(content="Generate Terraform module from execution plan")],
        active_agent="resource_configuration_agent",
        
        # Agent workspaces with planner data
        agent_workspaces={
            "resource_configuration_agent": {
                "planner_input": planner_output["resource_configurations"]  # ← Planner data
            },
            "variable_definition_agent": {
                "planner_input": planner_output["variable_definitions"]  # ← Planner data
            },
            "data_source_agent": {
                "planner_input": planner_output["data_sources"]  # ← Planner data
            },
            "local_values_agent": {
                "planner_input": planner_output["local_values"]  # ← Planner data
            }
        }
    )

# Generator Swarm coordinates 4 agents:
# 1. Resource Configuration Agent
# 2. Variable Definition Agent  
# 3. Data Source Agent
# 4. Local Values Agent
```

## 📊 State Transformation Details

### **Supervisor State → Generation State**

```python
@staticmethod
def supervisor_to_generation(supervisor_state: SupervisorState) -> GenerationState:
    """Transform supervisor state to generation state."""
    planner_data = supervisor_state.planner_data or {}  # ← Key extraction
    return GenerationState(
        requirements=planner_data.get("requirements_analysis", {}),
        provider_versions=planner_data.get("provider_versions", {}),
        registry_schemas_ref=supervisor_state.workspace_ref,
        standards_profile=planner_data.get("standards_profile", {}),
        module_name=f"module_{supervisor_state.workflow_id[:8]}",
    )
```

### **Planner Data Structure in Supervisor**

```python
# supervisor_state.planner_data contains:
{
    "execution_plan": {
        "service_name": "Amazon VPC",
        "module_name": "terraform-aws-vpc-advanced",
        "resource_configurations": [
            {
                "resource_address": "aws_vpc.this",
                "resource_type": "aws_vpc",
                "configuration": {
                    "cidr_block": "${var.vpc_ipv4_cidr_block}",
                    "enable_dns_support": "${var.enable_dns_support}",
                    "tags": "${merge(local.common_tags, { Name = local.vpc_name_tag })}"
                },
                "depends_on": [],
                "parameter_justification": "Supports either direct CIDR or IPAM..."
            }
        ],
        "variable_definitions": [
            {
                "name": "name_prefix",
                "type": "string",
                "description": "Base name used in Name tags for VPC and child resources",
                "validation_rules": [
                    "var.name_prefix == null || (length(var.name_prefix) >= 1 && length(var.name_prefix) <= 32)"
                ]
            }
        ],
        "data_sources": [
            {
                "resource_name": "available",
                "data_source_type": "aws_availability_zones",
                "configuration": {"state": "available"},
                "description": "Fetches available AZs to support multi-AZ deployments"
            }
        ],
        "local_values": [
            {
                "name": "vpc_name_tag",
                "expression": "var.name_prefix != null ? \"${var.name_prefix}-vpc\" : \"${var.name}-vpc\"",
                "description": "VPC Name tag value",
                "depends_on": ["local.name_base"]
            }
        ],
        "dependencies": [
            {
                "source_component": "aws_vpc.this",
                "target_component": "aws_internet_gateway.this",
                "dependency_type": "resource_dependency",
                "dependency_reason": "IGW requires VPC ID for attachment"
            }
        ],
        "security_considerations": [
            {
                "component": "aws_vpc.this",
                "security_impact": "medium",
                "considerations": [
                    "VPC CIDR blocks should not overlap with existing networks"
                ]
            }
        ],
        "cost_estimates": [
            {
                "component": "aws_nat_gateway.this",
                "estimated_monthly_cost": 45.60,
                "cost_breakdown": {
                    "nat_gateway_hours": 24 * 30 * 0.045
                }
            }
        ]
    }
}
```

## 🎯 Key Integration Points

### **1. Handoff Tools**

```python
# Supervisor creates handoff tools for all agents
handoff_tools = create_handoff_tools_for_agents([
    "planner_sub_supervisor",
    "generation_agent",  # ← This is the Generator Swarm
    "validation_agent",
    "editor_agent"
])

# Each handoff tool creates a Command with proper state updates
def handoff_to_generation_agent(task_description, state):
    return Command(
        goto="generation_agent",
        graph=Command.PARENT,
        update={
            "messages": messages + [tool_message],
            "active_agent": "generation_agent",
            "task_description": task_description,
            "user_request": user_request,
            "session_id": session_id,
            "task_id": task_id,
            "status": "in_progress"
        }
    )
```

### **2. State Merging**

```python
# When generator swarm completes, it returns results
# Supervisor merges results back using StateTransformer

if agent_type == AgentType.GENERATION:
    supervisor_updates = StateTransformer.generation_to_supervisor(GenerationState(**agent_state))
    # This creates:
    # {
    #     "generation_data": {
    #         "design_rationale": generation_state.design_rationale,
    #         "file_plan": generation_state.file_plan,
    #         "template_params": generation_state.template_params,
    #         "module_manifest": generation_state.module_manifest
    #     },
    #     "generated_module_ref": generation_state.generated_files_ref,
    #     "current_agent": AgentType.VALIDATION  # Next phase
    # }
```

### **3. Human-in-the-Loop Integration**

```python
# Both supervisor and generator swarm support HITL
# Supervisor handles high-level approvals
# Generator swarm handles tactical approvals

# Supervisor HITL
if supervisor_state.human_approval_required:
    yield AgentResponse(
        response_type='human_input',
        require_user_input=True,
        content=supervisor_state.approval_context.get('question', 'Input required')
    )

# Generator Swarm HITL (via generation_hitl.py)
if self.should_request_approval(context, "high_cost_resources"):
    await self.request_approval_high_cost_resources(
        approval_context={
            "resource_type": resource_config["type"],
            "estimated_monthly_cost": estimated_cost
        }
    )
```

## 🔧 Error Handling and Recovery

### **1. Supervisor Level**

```python
# Supervisor handles agent failures
if "error" in agent_state:
    updated_state_data.update({
        "error": agent_state["error"],
        "status": WorkflowStatus.FAILED,
        "current_agent": None
    })
```

### **2. Generator Swarm Level**

```python
# Generator swarm handles sub-agent failures
try:
    # Agent processing
    result = await agent.process()
except Exception as e:
    self.logger.log_structured(
        level="ERROR",
        message="Agent processing failed",
        extra={"error": str(e), "agent": agent_name}
    )
    # Handle error and potentially retry or escalate
```

## 📋 Summary

The complete flow from Supervisor Agent to Generator Swarm involves:

1. **Client Request** → **Supervisor Agent** (main orchestrator)
2. **Supervisor** → **Planner Sub-Supervisor** (strategic planning)
3. **Planner** → **Execution Plan** (comprehensive plan like `execution_planner_5.json`)
4. **Planner** → **Supervisor** (returns `planner_data`)
5. **Supervisor** → **Generator Swarm** (tactical implementation)
6. **Generator Swarm** → **Terraform Module** (complete module files)

### **Key Data Flow**:
- **Input**: Client request → Supervisor
- **Planning**: Supervisor → Planner → Execution Plan
- **Generation**: Execution Plan → Generator Swarm → Terraform Module
- **State Management**: `SupervisorState` with `planner_data` and `generation_data`
- **Coordination**: Handoff tools and StateTransformer classes
- **Human Oversight**: HITL at both supervisor and generator levels

This architecture provides a clean separation of concerns with the supervisor handling orchestration, the planner handling strategy, and the generator swarm handling tactical implementation.

---

*Last Updated: [Current Date]*
*Version: 1.0*
*Status: Ready for Implementation*
