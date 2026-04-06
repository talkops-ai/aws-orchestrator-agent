# Planner Sub Supervisor → Generator Swarm Input Specification

## 📋 Overview

This document defines the input structure that the **Planner Sub Supervisor** provides to the **Generator Swarm** for the Planning Stage of Terraform module generation.

## 🔄 Data Flow

```
┌─────────────────────────────────────┐
│        PLANNER SUB SUPERVISOR       │
│  ┌─────────────────────────────────┐ │
│  │  • Strategic Planning           │ │
│  │  • Resource Architecture        │ │
│  │  • Dependency Analysis          │ │
│  │  • Execution Plan Generation    │ │
│  └─────────────────────────────────┘ │
└─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│         GENERATOR SWARM             │
│  ┌─────────────────────────────────┐ │
│  │  • Resource Configuration       │ │
│  │  • Variable Definition          │ │
│  │  • Data Source Generation       │ │
│  │  • Local Values Creation        │ │
│  └─────────────────────────────────┘ │
└─────────────────────────────────────┘
```

## 📊 Input Structure

The Planner Sub Supervisor provides a comprehensive execution plan in the following JSON structure:

### Root Level Structure

```json
{
  "service_name": "Amazon VPC",
  "module_name": "terraform-aws-vpc-advanced", 
  "target_environment": "prod",
  "plan_generation_timestamp": "2025-09-02T00:00:00Z",
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

## 🗂️ Detailed Component Breakdown

### 1. **terraform_files** - File Organization Plan

```json
"terraform_files": [
  {
    "file_name": "main.tf",
    "file_purpose": "Main resource definitions",
    "resources_included": [
      "aws_vpc.this",
      "aws_internet_gateway.this",
      "aws_subnet.public"
    ],
    "dependencies": ["variables.tf", "locals.tf", "data.tf"],
    "organization_rationale": "Holds all managed AWS resources with explicit dependency ordering..."
  },
  {
    "file_name": "variables.tf", 
    "file_purpose": "Input variable declarations",
    "resources_included": [],
    "dependencies": [],
    "organization_rationale": "Declares all inputs with types, defaults, and validation..."
  }
]
```

**Purpose**: Defines the file structure and organization strategy for the Terraform module.

**Used by**: All agents to understand file organization and dependencies.

### 2. **variable_definitions** - Input Variable Specifications

```json
"variable_definitions": [
  {
    "name": "name_prefix",
    "type": "string", 
    "description": "Base name used in Name tags for VPC and child resources",
    "default": null,
    "sensitive": false,
    "nullable": true,
    "validation_rules": [
      "var.name_prefix == null || (length(var.name_prefix) >= 1 && length(var.name_prefix) <= 32)",
      "var.name_prefix == null || can(regex(\"^[a-z0-9-]+$\", var.name_prefix))"
    ],
    "example_values": ["app-prod-usw2", "platform-shared"],
    "justification": "Improves discoverability and consistency..."
  },
  {
    "name": "tags",
    "type": "map(string)",
    "description": "Global tags applied to all resources",
    "default": {},
    "sensitive": false,
    "nullable": false,
    "validation_rules": ["alltrue([for k, v in var.tags : length(trim(k)) > 0])"],
    "example_values": [{"Environment": "prod", "Project": "vpc"}],
    "justification": "Enables consistent resource tagging and governance..."
  }
]
```

**Purpose**: Defines all input variables with types, validation, and examples.

**Used by**: **Variable Definition Agent** to generate `variables.tf`.

### 3. **local_values** - Computed Value Specifications

```json
"local_values": [
  {
    "name": "checklist",
    "expression": "tolist([\"Finalize inputs and validations\",\"Resolve AZs and subnet plans\"])",
    "description": "Concise checklist of conceptual tasks executed by the module",
    "depends_on": [],
    "usage_context": "Documentation and planning confirmation"
  },
  {
    "name": "region", 
    "expression": "data.aws_region.current.name",
    "description": "Current AWS region",
    "depends_on": ["data.aws_region.current"],
    "usage_context": "Build service names for endpoints and naming conventions"
  },
  {
    "name": "vpc_name_tag",
    "expression": "var.name_prefix != null ? \"${var.name_prefix}-vpc\" : \"${var.name}-vpc\"",
    "description": "VPC Name tag value",
    "depends_on": ["local.name_base"],
    "usage_context": "VPC resource tagging"
  }
]
```

**Purpose**: Defines computed values and expressions for the module.

**Used by**: **Local Values Agent** to generate `locals.tf`.

### 4. **data_sources** - External Data Source Specifications

```json
"data_sources": [
  {
    "resource_name": "available",
    "data_source_type": "aws_availability_zones", 
    "configuration": {
      "state": "available"
    },
    "description": "Fetches available AZs to support multi-AZ deployments",
    "exported_attributes": ["names"],
    "error_handling": "Ensure at least two AZs are returned; otherwise, validation fails"
  },
  {
    "resource_name": "current",
    "data_source_type": "aws_region",
    "configuration": {},
    "description": "Determines current region for service names and S3 prefixes", 
    "exported_attributes": ["name"],
    "error_handling": "None; provider must be configured"
  },
  {
    "resource_name": "existing",
    "data_source_type": "aws_cloudwatch_log_group",
    "configuration": {
      "name": "var.vpc_flow_logs_cloudwatch_log_group_name"
    },
    "description": "References existing CloudWatch log group for VPC Flow Logs",
    "exported_attributes": ["arn", "id"],
    "error_handling": "If ARN invalid or key not found, Terraform will error..."
  }
]
```

**Purpose**: Defines external data sources and lookups.

**Used by**: **Data Source Agent** to generate `data.tf`.

### 5. **resource_configurations** - Resource Definition Specifications

```json
"resource_configurations": [
  {
    "resource_address": "aws_vpc.this",
    "resource_type": "aws_vpc",
    "resource_name": "this", 
    "configuration": {
      "cidr_block": "${var.ipv4_ipam_pool_id == null ? var.vpc_ipv4_cidr_block : null}",
      "ipv4_ipam_pool_id": "${var.ipv4_ipam_pool_id}",
      "assign_generated_ipv6_cidr_block": "${var.enable_ipv6 && var.assign_generated_ipv6_cidr_block}",
      "enable_dns_support": "${var.enable_dns_support}",
      "enable_dns_hostnames": "${var.enable_dns_hostnames}",
      "instance_tenancy": "${var.instance_tenancy}",
      "tags": "${merge(local.common_tags, { Name = local.vpc_name_tag })}"
    },
    "depends_on": [],
    "lifecycle_rules": {
      "prevent_destroy": false
    },
    "tags_strategy": "Merge global tags with Name tag; include ManagedBy and TerraformModule as required keys",
    "parameter_justification": "Supports either direct CIDR or IPAM; enables DNS features for service interoperability..."
  },
  {
    "resource_address": "aws_internet_gateway.this",
    "resource_type": "aws_internet_gateway", 
    "resource_name": "this",
    "configuration": {
      "vpc_id": "${aws_vpc.this.id}",
      "tags": "${merge(local.common_tags, var.igw_tags, { Name = local.igw_name_tag })}"
    },
    "depends_on": ["aws_vpc.this"],
    "lifecycle_rules": null,
    "tags_strategy": "Edge resource tagged for governance",
    "parameter_justification": "Created only when var.create_internet_gateway is true for public internet access..."
  }
]
```

**Purpose**: Defines all AWS resources with their configurations and dependencies.

**Used by**: **Resource Configuration Agent** to generate `main.tf`.

### 6. **outputs** - Output Value Specifications

```json
"outputs": [
  {
    "name": "vpc_id",
    "value": "aws_vpc.this.id",
    "description": "ID of the VPC",
    "sensitive": false,
    "depends_on": ["aws_vpc.this"]
  },
  {
    "name": "vpc_arn", 
    "value": "aws_vpc.this.arn",
    "description": "ARN of the VPC",
    "sensitive": false,
    "depends_on": ["aws_vpc.this"]
  }
]
```

**Purpose**: Defines output values exposed by the module.

**Used by**: All agents to understand what values are exported.

### 7. **dependencies** - Cross-Component Dependencies

```json
"dependencies": [
  {
    "source_component": "aws_vpc.this",
    "target_component": "aws_internet_gateway.this", 
    "dependency_type": "resource_dependency",
    "dependency_reason": "IGW requires VPC ID for attachment"
  },
  {
    "source_component": "data.aws_availability_zones.available",
    "target_component": "aws_subnet.public",
    "dependency_type": "data_dependency", 
    "dependency_reason": "Subnet creation requires AZ information"
  }
]
```

**Purpose**: Defines dependencies between different components.

**Used by**: **Generator State Controller** for coordination and **Handoff Manager** for dependency-aware handoffs.

### 8. **security_considerations** - Security Analysis

```json
"security_considerations": [
  {
    "component": "aws_vpc.this",
    "security_impact": "medium",
    "considerations": [
      "VPC CIDR blocks should not overlap with existing networks",
      "DNS settings affect service discovery and security"
    ],
    "recommendations": [
      "Use private subnets for sensitive workloads",
      "Enable VPC Flow Logs for network monitoring"
    ]
  }
]
```

**Purpose**: Provides security analysis and recommendations.

**Used by**: **HITL System** for security-critical approval decisions.

### 9. **cost_estimates** - Cost Analysis

```json
"cost_estimates": [
  {
    "component": "aws_nat_gateway.this",
    "estimated_monthly_cost": 45.60,
    "cost_breakdown": {
      "nat_gateway_hours": 24 * 30 * 0.045,
      "data_processing_gb": 100 * 0.045
    },
    "cost_optimization_tips": [
      "Consider NAT Instance for lower cost",
      "Use single NAT Gateway for multiple AZs if possible"
    ]
  }
]
```

**Purpose**: Provides cost estimates and optimization recommendations.

**Used by**: **HITL System** for high-cost approval decisions.

### 10. **deployment_strategy** - Deployment Planning

```json
"deployment_strategy": {
  "deployment_order": [
    "1. Create VPC and core networking",
    "2. Create subnets and routing", 
    "3. Create security groups and NACLs",
    "4. Create VPC endpoints and flow logs"
  ],
  "rollback_strategy": "Destroy resources in reverse order",
  "validation_checkpoints": [
    "Verify VPC CIDR doesn't conflict",
    "Confirm AZ availability",
    "Validate IAM permissions"
  ]
}
```

**Purpose**: Defines deployment strategy and validation checkpoints.

**Used by**: **Generator State Controller** for stage management.

## 🔄 Input Processing in Generator Swarm

### 1. **Input Transformation**

The Generator Swarm receives the planner output and transforms it into the `GeneratorStageState`:

```python
def transform_planner_input_to_state(planner_output: Dict[str, Any]) -> GeneratorStageState:
    """Transform planner output into GeneratorStageState"""
    
    return GeneratorStageState(
        # Core swarm fields
        messages=[HumanMessage(content="Generate Terraform module from execution plan")],
        active_agent="resource_configuration_agent",
        
        # Planning stage management
        stage_status="planning_active",
        planning_progress={
            "resource_configuration_agent": 0.0,
            "variable_definition_agent": 0.0, 
            "data_source_agent": 0.0,
            "local_values_agent": 0.0
        },
        
        # Agent workspaces with planner data
        agent_workspaces={
            "resource_configuration_agent": {
                "generated_resources": [],
                "pending_variable_requests": [],
                "pending_data_source_requests": [],
                "completion_checklist": [],
                "planner_input": planner_output["resource_configurations"]  # ← Planner data
            },
            "variable_definition_agent": {
                "generated_variables": [],
                "variable_validation_rules": [],
                "source_requests": [],
                "completion_checklist": [],
                "planner_input": planner_output["variable_definitions"]  # ← Planner data
            },
            "data_source_agent": {
                "generated_data_sources": [],
                "external_dependencies": [],
                "completion_checklist": [],
                "planner_input": planner_output["data_sources"]  # ← Planner data
            },
            "local_values_agent": {
                "generated_locals": [],
                "computed_expressions": [],
                "completion_checklist": [],
                "planner_input": planner_output["local_values"]  # ← Planner data
            }
        },
        
        # Dependency tracking from planner
        dependency_graph=build_dependency_graph(planner_output["dependencies"]),
        
        # HITL context from planner
        approval_context={
            "security_considerations": planner_output["security_considerations"],
            "cost_estimates": planner_output["cost_estimates"]
        }
    )
```

### 2. **Agent-Specific Data Access**

Each agent accesses its specific planner data:

```python
# Resource Configuration Agent
def generate_terraform_resources(state: GeneratorStageState) -> str:
    planner_data = state["agent_workspaces"]["resource_configuration_agent"]["planner_input"]
    
    for resource_config in planner_data:
        # Process each resource configuration from planner
        resource_code = create_resource_code(resource_config)
        # Add to generated resources
        state["agent_workspaces"]["resource_configuration_agent"]["generated_resources"].append(resource_code)

# Variable Definition Agent  
def generate_terraform_variables(state: GeneratorStageState) -> str:
    planner_data = state["agent_workspaces"]["variable_definition_agent"]["planner_input"]
    
    for variable_def in planner_data:
        # Process each variable definition from planner
        variable_code = create_variable_code(variable_def)
        # Add to generated variables
        state["agent_workspaces"]["variable_definition_agent"]["generated_variables"].append(variable_code)
```

### 3. **Dependency Resolution**

The planner's dependency information is used for agent coordination:

```python
def build_dependency_graph(planner_dependencies: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    """Build dependency graph from planner dependencies"""
    
    dependency_graph = {}
    
    for dep in planner_dependencies:
        source = dep["source_component"]
        target = dep["target_component"]
        
        # Map to agent names
        source_agent = map_component_to_agent(source)
        target_agent = map_component_to_agent(target)
        
        if target_agent not in dependency_graph:
            dependency_graph[target_agent] = set()
        dependency_graph[target_agent].add(source_agent)
    
    return dependency_graph

def map_component_to_agent(component: str) -> str:
    """Map component to responsible agent"""
    if component.startswith("aws_"):
        return "resource_configuration_agent"
    elif component.startswith("var."):
        return "variable_definition_agent"
    elif component.startswith("data."):
        return "data_source_agent"
    elif component.startswith("local."):
        return "local_values_agent"
    else:
        return "resource_configuration_agent"  # Default
```

## 🎯 Key Integration Points

### 1. **Resource Configuration Agent**
- **Input**: `resource_configurations` array
- **Processing**: Creates Terraform resource blocks
- **Output**: `main.tf` content
- **Dependencies**: Uses variables and data sources from other agents

### 2. **Variable Definition Agent**
- **Input**: `variable_definitions` array  
- **Processing**: Creates variable declarations with validation
- **Output**: `variables.tf` content
- **Dependencies**: May depend on local values for complex validation

### 3. **Data Source Agent**
- **Input**: `data_sources` array
- **Processing**: Creates data source blocks
- **Output**: `data.tf` content
- **Dependencies**: May depend on variables for configuration

### 4. **Local Values Agent**
- **Input**: `local_values` array
- **Processing**: Creates local value blocks with expressions
- **Output**: `locals.tf` content
- **Dependencies**: Uses variables and data sources

## 🔧 HITL Integration Points

### 1. **Security-Critical Approvals**
```python
# From security_considerations in planner input
if resource_config["security_impact"] in ["high", "critical"]:
    await self.request_approval_security_critical(
        approval_context={
            "resource_type": resource_config["resource_type"],
            "security_impact": resource_config["security_impact"],
            "considerations": resource_config["security_considerations"]
        }
    )
```

### 2. **High-Cost Approvals**
```python
# From cost_estimates in planner input
if resource_config["estimated_monthly_cost"] > 1000:
    await self.request_approval_high_cost_resources(
        approval_context={
            "resource_type": resource_config["resource_type"],
            "estimated_monthly_cost": resource_config["estimated_monthly_cost"],
            "cost_breakdown": resource_config["cost_breakdown"]
        }
    )
```

## 📋 Summary

The Planner Sub Supervisor provides a comprehensive execution plan that includes:

1. **Complete Resource Specifications** - All AWS resources with configurations
2. **Variable Definitions** - Input variables with validation and examples  
3. **Data Source Specifications** - External data lookups and references
4. **Local Value Expressions** - Computed values and complex expressions
5. **Dependency Information** - Cross-component dependencies for coordination
6. **Security Analysis** - Security considerations for HITL approvals
7. **Cost Estimates** - Cost analysis for HITL approvals
8. **File Organization** - Terraform file structure and organization strategy

This rich input enables the Generator Swarm to create a complete, production-ready Terraform module with proper coordination between agents and human oversight for critical decisions.

---

*Last Updated: [Current Date]*
*Version: 1.0*
*Status: Ready for Implementation*
