class DefaultConfig:
    """Default configuration for the AWS Orchestrator Agent."""
    # LLM Configuration
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 12000
    
    # Higher LLM Configuration (for complex reasoning tasks)
    LLM_HIGHER_PROVIDER: str = "openai"
    LLM_HIGHER_MODEL: str = "gpt-5-mini"
    LLM_HIGHER_TEMPERATURE: float = 1.0
    LLM_HIGHER_MAX_TOKENS: int = 15000

    LLM_REACT_AGENT_PROVIDER: str = "openai"
    LLM_REACT_AGENT_MODEL: str = "gpt-4.1-mini"
    LLM_REACT_AGENT_TEMPERATURE: float = 0.0
    LLM_REACT_AGENT_MAX_TOKENS: int = 12000

    # Module Configuration
    MODULE_PATH: str = "/Users/structbinary/Documents/work/talkops/aws-orchestrator-agent"
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "aws_orchestrator_agent.log"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
    LOG_TO_CONSOLE: bool = True
    LOG_TO_FILE: bool = True
    LOG_STRUCTURED_JSON: bool = False
    
    # MCP Server Configuration
    TERRAFORM_MCP_SERVER_HOST: str = "localhost"
    TERRAFORM_MCP_SERVER_PORT: int = 8000
    TERRAFORM_MCP_SERVER_TRANSPORT: str = "sse"
    TERRAFORM_MCP_SERVER_DISABLED: bool = False
    AGENTS_MCP_SERVER_AUTO_APPROVE: list = []
    
    # Supervisor Agent Configuration
    SUPERVISOR_OUTPUT_MODE: str = "full_history"
    SUPERVISOR_ADD_HANDOFF_BACK_MESSAGES: bool = True
    SUPERVISOR_MAX_SNAPSHOTS: int = 10
    SUPERVISOR_ENABLE_AUDIT_TRAIL: bool = True
    SUPERVISOR_MAX_CONCURRENT_WORKFLOWS: int = 10
    SUPERVISOR_WORKFLOW_TIMEOUT: int = 300
    SUPERVISOR_MAX_RETRIES: int = 3
    SUPERVISOR_TIMEOUT_SECONDS: int = 300
    SUPERVISOR_HUMAN_APPROVAL_REQUIRED: list = [
        "terraform_apply",
        "security_violations", 
        "cost_threshold_exceeded",
        "breaking_changes",
        "production_deployment"
    ]
    SUPERVISOR_VALIDATION_ALWAYS_REQUIRED: list = [
        "terraform_generation",
        "terraform_modification",
        "infrastructure_changes"
    ]
    
    # Agent Names and Descriptions
    SUPERVISOR_AGENT_NAMES: dict = {
        "analysis": "Analysis Agent",
        "generation": "Generation Agent",
        "validation": "Validation Agent", 
        "editor": "Editor Agent"
    }
    
    SUPERVISOR_AGENT_DESCRIPTIONS: dict = {
        "analysis": "Handles requirements analysis, conversation management, and AWS context retrieval",
        "generation": "Creates new Terraform modules from scratch with best practices and patterns",
        "validation": "Performs comprehensive validation including syntax, plan, security, and compliance checks",
        "editor": "Modifies existing Terraform configurations with surgical precision and minimal disruption"
    }
    
    # Routing Rules Configuration
    SUPERVISOR_ROUTING_RULES: dict = {
        "analysis": [
            "requirements_gathering",
            "conversation_management", 
            "context_retrieval",
            "user_clarification",
            "workflow_planning"
        ],
        "generation": [
            "new_terraform_modules",
            "infrastructure_generation",
            "best_practices_implementation",
            "pattern_application"
        ],
        "validation": [
            "terraform_validation",
            "security_scanning",
            "compliance_checks",
            "plan_validation",
            "cost_analysis"
        ],
        "editor": [
            "terraform_modification",
            "surgical_updates",
            "version_compatibility",
            "state_management"
        ]
    }
    
    # Error Handling Configuration
    SUPERVISOR_ERROR_HANDLING_CONFIG: dict = {
        "max_retries_per_agent": 2,
        "retry_delay_seconds": 5,
        "escalation_threshold": 3,
        "fallback_agents": {
            "generation": "editor",
            "validation": "analysis",
            "editor": "generation"
        }
    }
    
    # State Schema Configuration
    SUPERVISOR_STATE_SCHEMA_CONFIG: dict = {
        "shared_keys": ["messages", "current_workflow", "user_approval_required"],
        "supervisor_keys": ["supervisor_context", "routing_history", "error_context"],
        "subgraph_keys": {
            "analysis": ["analysis_context", "requirements", "aws_context"],
            "generation": ["generation_context", "terraform_code", "mcp_registry_data"],
            "validation": ["validation_context", "validation_results", "validation_stages"],
            "editor": ["editor_context", "modifications", "state_analysis"]
        }
    }
    
    # A2A Server Configuration
    A2A_SERVER_HOST: str = "localhost"
    A2A_SERVER_PORT: int = 10102
    
    # Configuration Optimizer Default Values
    CONFIG_OPTIMIZER_ENVIRONMENT: str = "prod"
    CONFIG_OPTIMIZER_EXPECTED_LOAD: str = "high"
    CONFIG_OPTIMIZER_BUDGET_CONSTRAINTS: str = "Optimize TCO; minimize data transfer costs; ensure high availability; right-size resources."
    CONFIG_OPTIMIZER_COMPLIANCE_REQUIREMENTS: list = ["SOC2-Type-II", "ISO-27001", "GDPR", "CCPA", "PCI-DSS"]
    CONFIG_OPTIMIZER_OPTIMIZATION_TARGETS: list = ["security", "reliability", "cost", "performance", "operational_excellence"]
    CONFIG_OPTIMIZER_ORGANIZATION_STANDARDS: dict = {
        "naming_conventions": ["lowercase-hyphen", "$<app>-$<env>-$<region>"],
        "tagging": {
            "required_keys": ["Name", "Environment", "Owner", "Application", "CostCenter", "DataClassification", "ManagedBy", "TerraformModule"]
        },
        "encryption": {"at_rest": "KMS-CMK-required", "in_transit": "TLS1.2+"},
        "logging": {"centralized": "CloudWatch/S3", "retention_days": 365},
        "availability": {"multi_az": True, "backup_strategy": "automated", "disaster_recovery": "enabled"},
        "compliance": {"data_residency": "region-locked", "pii_handling": "least-privilege"},
        "iac": {"terraform_version": ">=1.5", "providers_pinned": True, "code_review": "required"}
    }
    # State Management Configuration
    STATE_MGMT_DEFAULT_INFRASTRUCTURE_SCALE: str = "medium"
    STATE_MGMT_DEFAULT_ENVIRONMENTS: list = ["dev", "staging", "prod"]
    STATE_MGMT_DEFAULT_AWS_REGION: str = "us-east-1"
    STATE_MGMT_DEFAULT_MULTI_REGION: bool = False
    STATE_MGMT_DEFAULT_TEAM_SIZE: str = "small"
    STATE_MGMT_DEFAULT_TEAMS: list = ["platform", "development", "operations"]
    STATE_MGMT_DEFAULT_CONCURRENT_OPERATIONS: str = "low"
    STATE_MGMT_DEFAULT_CI_CD_INTEGRATION: str = "GitHub Actions"
    STATE_MGMT_DEFAULT_ENCRYPTION_REQUIRED: bool = True
    STATE_MGMT_DEFAULT_AUDIT_LOGGING: bool = True
    STATE_MGMT_DEFAULT_BACKUP_RETENTION_DAYS: int = 30
    STATE_MGMT_DEFAULT_COMPLIANCE_STANDARDS: list = ["SOC2-Type-II", "ISO-27001"]