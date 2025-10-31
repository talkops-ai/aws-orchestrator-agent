# AWS Orchestrator Agent

<div align="center">

[![Discord](https://img.shields.io/badge/Discord-Join%20Community-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/2V8AAufgp6)
[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-sandeep2014/aws--orchestrator--agent-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://hub.docker.com/r/sandeep2014/aws-orchestrator-agent)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-FF6B6B?style=for-the-badge)](https://github.com/langchain-ai/langgraph)
[![A2A Protocol](https://img.shields.io/badge/Google%20A2A-Protocol-4285F4?style=for-the-badge)](https://github.com/google/a2a)

</div>

---

## 🚀 Enterprise-Grade Multi-Agent Terraform Module Generation System

**AWS Orchestrator Agent** is a sophisticated, autonomous multi-agent system that generates enterprise-level AWS Terraform modules through intelligent research, deep analysis, and AI-powered code generation. Built with LangGraph and the Google A2A protocol, it leverages advanced AI orchestration patterns to deliver production-ready infrastructure automation through coordinated specialist agents.

## 🏗️ Architecture Overview

### Complete Multi-Agent Ecosystem

```mermaid
graph TD
    %% User Interface
    U["👤 User Request<br/>\"Help me create an enterprise S3 module\""] --> S["🎯 Main Supervisor Agent<br/>Orchestrates autonomous research & generation"]
    
    %% Main Supervisor
    S --> P["📋 Planner Sub-Supervisor<br/>Deep research & enterprise-grade execution plans"]
    S --> G["🏭 Generator Swarm<br/>Autonomous code generation using 7 specialized agents"]
    S --> W["📝 Writer React Agent<br/>Writes enterprise modules to filesystem"]
    S --> V["✅ Validation Agent<br/>Enterprise-grade validation & compliance"]
    
    %% Planner Sub-Agents
    subgraph PlannerAgents ["📋 Planner Sub-Supervisor Agents"]
        RA["🔍 Requirements Analyzer<br/>Extracts infrastructure requirements"]
        EP["📊 Execution Planner<br/>Creates detailed execution plans"]
        SP["🔒 Security & Best Practices<br/>Evaluates security & compliance"]
    end
    
    %% Generator Swarm Agents
    subgraph GeneratorAgents ["🏭 Generator Swarm Agents"]
        RC["🏗️ Resource Configuration<br/>Generates Terraform resources"]
        VD["📝 Variable Definition<br/>Creates variable definitions"]
        DS["🔍 Data Source<br/>Generates data source blocks"]
        LV["💾 Local Values<br/>Creates local value blocks"]
        OD["📤 Output Definition<br/>Generates output definitions"]
        BG["🗄️ Backend Generator<br/>Creates backend configuration"]
        RG["📚 README Generator<br/>Generates documentation"]
    end
    
    %% Writer Tools
    subgraph WriterTools ["📝 Writer React Agent Tools"]
        WF["📄 Write Files<br/>Writes Terraform files to disk"]
        BF["📦 Batch Write<br/>Efficient batch file operations"]
        VF["✅ Validate Syntax<br/>Validates HCL before writing"]
        CF["📁 Create Directories<br/>Manages directory structure"]
    end
    
    %% Data Flow
    P --> RA
    P --> EP
    P --> SP
    RA --> EP
    EP --> SP
    
    G --> RC
    RC --> VD
    RC --> DS
    RC --> LV
    RC --> OD
    VD --> BG
    DS --> BG
    LV --> BG
    OD --> BG
    BG --> RG
    
    W --> WF
    W --> BF
    W --> VF
    W --> CF
    
    %% Return Flow
    P --> S
    G --> S
    W --> S
    V --> S
    S --> U

    %% Human-in-the-Loop
    subgraph HumanLoop ["🔄 Human-in-the-Loop"]
        H["👤 User Clarification<br/>Interactive requirements gathering"]
    end
    
    S -.->|"Clarification needed"| H
    H -.->|"Refined requirements"| S
    
    %% Styling
    classDef user fill:#e1f5fe
    classDef supervisor fill:#f3e5f5
    classDef planner fill:#e8f5e8
    classDef generator fill:#fff3e0
    classDef writer fill:#fce4ec
    classDef validation fill:#f0f4c3
    classDef human fill:#ffebee
    
    class U user
    class S supervisor
    class P,RA,EP,SP planner
    class G,RC,VD,DS,LV,OD,BG,RG generator
    class W,WF,BF,VF,CF writer
    class V validation
    class H human
```

## 🔗 A2A Protocol Integration

### Enterprise Agent-to-Agent Communication

The AWS Orchestrator Agent is designed as a **first-class A2A (Agent-to-Agent) protocol agent**, enabling seamless integration with enterprise agent ecosystems and multi-agent coordination.

#### A2A Architecture Integration

```mermaid
graph TD
    A[A2A Request] --> B[DefaultRequestHandler]
    B --> C[AWSOrchestratorAgentExecutor]
    C --> D[SupervisorAgentAdapter]
    D --> E[SupervisorAgent]
    E --> F[StateManager]
    F --> G[Agent Subgraphs]
    
    G --> H[Analysis Agent]
    G --> I[Generation Agent]
    G --> J[Validation Agent]
    G --> K[Editor Agent]
    
    style D fill:#e1f5fe
    style E fill:#f3e5f5
    style F fill:#e8f5e8
```

#### Key A2A Components

1. **🔗 A2A Executor Integration**: `AWSOrchestratorAgentExecutor` implements the A2A protocol
2. **🔄 Supervisor Agent Adapter**: Bridges LangGraph-based Supervisor with A2A protocol
3. **🌐 A2A Server Integration**: Deployable as an A2A agent server
4. **📊 State Management**: Enterprise-grade workflow state tracking across agent boundaries

#### A2A Server Deployment

```bash
# Development A2A Server (using uv)
uv run --active aws-orchestrator-agent \
  --host localhost \
  --port 10102 \
  --agent-card aws_orchestrator_agent/card/aws_orchestrator_agent.json

# Production A2A Server
aws-orchestrator-agent \
  --host 0.0.0.0 \
  --port 8000 \
  --agent-card aws_orchestrator_agent/card/aws_orchestrator_agent.json \
  --config-file config.yaml
```

## 🎯 Core Components

### 1. 🎯 Main Supervisor Agent (`CustomSupervisorAgent`)

**The Autonomous Orchestrator** - Manages the entire workflow lifecycle using `langgraph-supervisor`.

#### Key Features:
- **🔄 Autonomous Workflow Orchestration**: Coordinates all sub-agents through intelligent routing
- **📊 Comprehensive State Management**: Maintains detailed workflow state across all phases
- **🛡️ Robust Error Handling**: Advanced error recovery and retry mechanisms
- **👤 Human-in-the-Loop**: Interactive clarification for complex requirements[To be release on next version]
- **📈 Real-Time Progress Tracking**: Live workflow progress monitoring and reporting
- **🔀 Intelligent Agent Routing**: Smart task delegation based on requirements
- **⏱️ Time-Intensive Processing**: Designed for thorough, high-quality module generation

#### Autonomous Workflow Phases:
1. **📋 Deep Research Phase**: Delegates to Planner Sub-Supervisor for comprehensive analysis
2. **🏭 Autonomous Generation Phase**: Delegates to Generator Swarm for production-grade code generation
3. **📝 Writing Phase**: Delegates to Writer React Agent for production-ready file creation

### 2. 📋 Planner Sub-Supervisor Agent

**The Research Engine** - Conducts deep research and analysis to create comprehensive execution plans.

#### Sub-Agents:
- **🔍 Requirements Analyzer**: Deep analysis of infrastructure requirements
- **📊 Execution Planner**: Creates detailed execution plans with comprehensive resource configurations
- **🔒 Security & Best Practices Evaluator**: Ensures security standards and compliance (optional)

#### Output:
- **Comprehensive Requirements Data**: Detailed infrastructure requirements
- **Execution Plans**: Production-ready Terraform module specifications
- **Advanced Resource Configurations**: Complete resource definitions with best practices
- **Variable Definitions**: Comprehensive input variable specifications
- **Module Structure**: Production-ready file organization and architecture

### 3. 🏭 Generator Swarm Agent

**The Autonomous Code Generation Engine** - Uses 7 specialized agents to generate production-grade Terraform modules through sophisticated coordination.

#### Specialized Agents:
1. **🏗️ Resource Configuration Agent**: Generates Terraform resource blocks with best practices
2. **📝 Variable Definition Agent**: Creates comprehensive variable definitions with validation
3. **🔍 Data Source Agent**: Generates advanced data source blocks
4. **💾 Local Values Agent**: Creates sophisticated local value blocks for complex logic
5. **📤 Output Definition Agent**: Generates output definitions with proper documentation
6. **🗄️ Backend Generator Agent**: Creates production-ready backend configuration
7. **📚 README Generator Agent**: Generates comprehensive documentation

#### Features:
- **🔄 Sophisticated Dependency-Aware Handoffs**: Advanced inter-agent coordination
- **📊 Isolated State Management**: Separate state schemas for each agent to prevent conflicts
- **🎯 Priority-Based Routing**: Handoffs based on dependency priority
- **🛡️ Error Recovery**: Individual agent error handling with graceful continuation
- **📈 Real-Time Progress Tracking**: Live generation progress monitoring
- **⏱️ Time-Intensive Processing**: Designed for thorough, high-quality module generation

### 4. 📝 Writer React Agent

**The File System Manager** - Writes Terraform modules to the filesystem with production-ready organization.

#### Tools:
- **📄 File Writing**: Individual file writing with validation
- **📦 Batch Operations**: Efficient batch file operations
- **✅ Syntax Validation**: HCL syntax validation with best practices
- **📁 Directory Management**: Production-ready directory structure management
- **📋 File Management**: Advanced directory listing and file management
- **📖 File Analysis**: Comprehensive file content reading and analysis

#### Features:
- **🔄 React Agent Pattern**: Tool-based execution with state injection
- **📊 Operation Tracking**: Detailed file operation logging
- **🛡️ Error Handling**: Individual file error handling with graceful continuation
- **💾 Backup Support**: Automatic backup creation for file operations
- **📈 Progress Monitoring**: Real-time writing progress tracking

### 5. ✅ Validation Agent [Not Integrated - To be released on next version]

**The Quality Assurance Engine** - Validates generated Terraform modules for correctness and best practices.

#### Features:
- **🔍 Syntax Validation**: Comprehensive HCL syntax checking
- **📋 Best Practices**: AWS and Terraform best practices validation
- **🔒 Security Scanning**: Advanced security vulnerability detection and compliance
- **📊 Resource Validation**: Comprehensive resource configuration validation
- **🎯 Compliance Checking**: Regulatory compliance verification

## 🔄 Complete Workflow Example

### Enterprise S3 Module Generation Flow

```mermaid
sequenceDiagram
    participant U as 👤 User
    participant S as 🎯 Supervisor
    participant P as 📋 Planner
    participant G as 🏭 Generator Swarm
    participant W as 📝 Writer
    participant FS as 💾 File System
    
    U->>S: "Help me create an enterprise S3 module"
    S->>P: Delegate to Planner Sub-Supervisor
    
    Note over P: Deep Enterprise Research & Analysis
    P->>P: Conduct comprehensive S3 requirements research
    P->>P: Analyze enterprise patterns and best practices
    P->>P: Create detailed enterprise execution plan
    P->>S: Return comprehensive planning data
    
    S->>G: Delegate to Generator Swarm
    
    Note over G: Autonomous Enterprise Code Generation
    G->>G: Generate enterprise S3 bucket resource with best practices
    G->>G: Generate comprehensive variables with validation
    G->>G: Generate enterprise outputs with documentation
    G->>G: Generate enterprise documentation and README
    G->>S: Return enterprise generation data
    
    S->>W: Delegate to Writer Agent
    
    Note over W: Enterprise File Writing
    W->>W: Validate enterprise HCL syntax
    W->>W: Create enterprise directory structure
    W->>FS: Write enterprise main.tf
    W->>FS: Write enterprise variables.tf
    W->>FS: Write enterprise outputs.tf
    W->>FS: Write enterprise README.md
    W->>S: Return enterprise completion status
    
    S->>U: "Enterprise S3 module created successfully!"
    
    Note over U,S: ⏱️ Time-Intensive Process: 20-25 minutes for enterprise-grade modules
```

## 🚀 Key Benefits

### 1. **🎯 Autonomous Multi-Agent Orchestration**
- **Intelligent Task Delegation**: Automatic routing to specialized agents based on workflow requirements
- **State-Driven Workflows**: Comprehensive state tracking across planning, generation, and writing phases
- **Robust Error Recovery**: Individual agent error handling with automatic retry mechanisms
- **Real-Time Progress Tracking**: Live monitoring and reporting throughout the generation process
- **Thorough Processing**: 20-25 minute generation time ensures high-quality, production-ready output

### 2. **🔗 A2A Protocol Integration**
- **Agent Ecosystem Compatibility**: Seamless integration with other A2A-compliant agents
- **Cross-Agent Coordination**: Work alongside security, compliance, and monitoring agents
- **Protocol Standardization**: Built on Google's A2A protocol for interoperability *(Note: Authentication/Authorization not yet implemented)*
- **Scalable Architecture**: Support for multiple concurrent workflows with isolated state management
- **Distributed State Orchestration**: Robust state management across agent boundaries

### 3. **🏗️ Modular Architecture**
- **Specialized Agent Roles**: 7+ specialized agents (Planner, Generator Swarm, Writer, etc.) with focused responsibilities
- **State Isolation**: Separate state schemas prevent conflicts and ensure agent independence
- **Tool-Based Execution**: ReAct pattern with sophisticated tool integration for file operations
- **Easy Extensibility**: Clear interfaces for adding new agents, tools, or capabilities

### 4. **🔄 Advanced Agent Coordination**
- **Dependency-Aware Handoffs**: Inter-agent communication based on workflow dependencies
- **Priority-Based Routing**: Intelligent handoff sequencing for optimal generation flow
- **State Transformation**: Automatic state adaptation between different agent contexts
- **Completion Detection**: Smart detection and reporting of workflow completion

### 5. **🛡️ Production-Ready Features**
- **Comprehensive Error Handling**: Multi-level error handling from supervisor to individual tools
- **Structured Logging**: Detailed logs throughout the system for debugging and monitoring
- **HCL Syntax Validation**: Built-in validation before writing Terraform files
- **Backup Support**: Automatic file backups during write operations
- **Human-in-the-Loop Ready**: Framework prepared for interactive clarification workflows *(Coming in next version)*

### 6. **📊 Complete Terraform Modules**
- **Full Module Structure**: Generates all necessary files (main.tf, variables.tf, outputs.tf, backend.tf, data.tf, locals.tf)
- **Comprehensive Documentation**: Auto-generated README with usage examples, inputs, outputs, and best practices
- **Best Practices Built-In**: Security configurations, naming conventions, and AWS best practices
- **Ready to Deploy**: Validated syntax and organized file structure for immediate use

## 🔧 Configuration & Customization

### Environment Configuration

AWS Orchestrator Agent can be customized via environment variables or a `.env` file in the project root.

Below are the main configuration options with their default values:

```python
# aws_orchestrator_agent/config/default.py

# LLM Configuration
LLM_PROVIDER = "openai"                    # LLM provider (openai, anthropic, azure, etc.)
LLM_MODEL = "gpt-4o-mini"                  # Model name for standard operations
LLM_TEMPERATURE = 0.0                      # Sampling temperature (0.0-1.0)
LLM_MAX_TOKENS = 15000                     # Maximum tokens per response

# Higher LLM Configuration (for complex reasoning tasks)
LLM_HIGHER_PROVIDER = "openai"             # Provider for complex reasoning
LLM_HIGHER_MODEL = "gpt-5-mini"            # Model for complex tasks
LLM_HIGHER_TEMPERATURE = 0.0               # Temperature for complex reasoning
LLM_HIGHER_MAX_TOKENS = 15000              # Max tokens for complex tasks

# React Agent LLM Configuration (for Writer Agent)
LLM_REACT_AGENT_PROVIDER = "openai"        # Provider for file operations
LLM_REACT_AGENT_MODEL = "gpt-4.1-mini"     # Model for Writer React Agent
LLM_REACT_AGENT_TEMPERATURE = 0.0          # Temperature for file operations
LLM_REACT_AGENT_MAX_TOKENS = 25000         # Max tokens for file operations

# Module Configuration
MODULE_PATH = "/path/to/aws-orchestrator-agent"  # Project root path

# Logging Configuration
LOG_LEVEL = "INFO"                         # Logging level (DEBUG, INFO, WARNING, ERROR)
LOG_FILE = "aws_orchestrator_agent.log"    # Log file path
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_TO_CONSOLE = True                      # Output logs to console
LOG_TO_FILE = True                         # Output logs to file
LOG_STRUCTURED_JSON = False                # Output logs in JSON format


# Supervisor Agent Configuration
SUPERVISOR_OUTPUT_MODE = "full_history"    # Output mode for supervisor
SUPERVISOR_MAX_RETRIES = 3                 # Max retry attempts
SUPERVISOR_TIMEOUT_SECONDS = 300           # Workflow timeout (seconds)
SUPERVISOR_MAX_CONCURRENT_WORKFLOWS = 10   # Max concurrent workflows

# A2A Server Configuration
A2A_SERVER_HOST = "localhost"              # A2A server host
A2A_SERVER_PORT = 10102                    # A2A server port
```

> **💡 Tip:**  
> You can override any of these settings by adding them to your `.env` file or exporting them as environment variables before starting the AWS Orchestrator Agent.

**Example `.env` file:**

```env
# LLM Configuration
OPENAI_API_KEY=sk-your-openai-api-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=20000

# Logging
LOG_LEVEL=DEBUG
LOG_TO_CONSOLE=True
LOG_STRUCTURED_JSON=False


# A2A Server
A2A_SERVER_HOST=0.0.0.0
A2A_SERVER_PORT=10102

# Module Path
MODULE_PATH=/Users/yourname/projects/aws-orchestrator-agent
```

> **📖 Note:**  
> For a complete list of all configuration options, security settings, compliance requirements, and advanced supervisor configurations, see [`aws_orchestrator_agent/config/default.py`](aws_orchestrator_agent/config/default.py).

---

### LLM Configuration

#### Multi-Provider Architecture
The system uses a modular, extensible LLM provider architecture built on an abstract base class pattern:

- **Multi-Provider Support**: Support for multiple LLM providers (Anthropic, OpenAI, Azure, etc.)
- **Model Selection**: Configurable model selection per agent
- **Parameter Tuning**: Adjustable temperature, max tokens, and other parameters
- **Provider Switching**: Easy switching between different LLM providers

#### Adding a New LLM Provider

To add a new LLM provider to the system, follow these steps:

##### 1. Implement the Provider Class
**Location:** `aws_orchestrator_agent/core/llm/llm_provider.py`

Create a new provider class that inherits from `BaseLLMProvider`:

```python
from .base_llm_provider import BaseLLMProvider
from langchain_core.runnables import Runnable

class MyNewProvider(BaseLLMProvider):
    """
    Concrete LLM provider for MyNewProvider models.
    Implements the BaseLLMProvider interface.
    """
    def create_llm(
        self,
        model: str,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        timeout: int = 60,
        **kwargs: Any
    ) -> Runnable:
        # Import your provider's SDK or LangChain integration
        # from my_provider_sdk import MyProviderLLM
        
        # Build configuration dictionary
        config = {
            "model": model,
            "temperature": temperature,
            # ... other parameters ...
        }
        config.update(kwargs)
        
        # Return a LangChain-compatible Runnable
        return MyProviderLLM(**config)
```

##### 2. Register in the Factory
Update the `_create_provider_instance` method in `LLMProvider`:

```python
elif provider == "mynewprovider":
    return MyNewProvider().create_llm(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        **kwargs
    )
```

Add your provider to the `_SUPPORTED_PROVIDERS` set.

##### 3. Add Environment Variables (Optional)
If your provider requires API keys or endpoints, add them to your `.env` file:
```bash
MY_NEW_PROVIDER_API_KEY=your_api_key_here
MY_NEW_PROVIDER_ENDPOINT=https://api.example.com
```

##### 4. Test Your Provider
Ensure your provider works by running the agent with your provider selected in the configuration.

#### Best Practices for Provider Implementation
- Follow the structure and docstring style of existing providers
- Use type hints and clear error messages
- Keep all provider-specific logic encapsulated in your provider class
- Do not modify agent or planner code—only the provider and factory
- Handle API keys and endpoints securely

#### Reference Documentation
- [Base LLM Provider](aws_orchestrator_agent/core/llm/base_llm_provider.py)
- [LLM Provider Factory](aws_orchestrator_agent/core/llm/llm_provider.py)
- [Complete Onboarding Guide](docs/ONBOARDING_LLM_PROVIDER.md)

### Agent Configuration
- **Custom Prompts**: Configurable prompts for each agent
- **Tool Selection**: Selective tool enabling/disabling
- **Workflow Customization**: Customizable workflow phases and routing
- **Error Handling**: Configurable error handling and retry policies

## 🎯 Use Cases

### 1. **🏗️ Enterprise Infrastructure as Code**
- Generate complete enterprise-grade Terraform modules for AWS services
- Create standardized, reusable enterprise infrastructure components
- Ensure enterprise best practices and security compliance
- Automate enterprise infrastructure provisioning workflows
- **⏱️ Time-Intensive**: 20-25 minutes per enterprise module for thorough analysis and generation

### 2. **🔄 DevOps Automation**
- Integrate with enterprise CI/CD pipelines
- Automate enterprise infrastructure testing and validation
- Streamline enterprise deployment processes
- Reduce manual enterprise infrastructure management
- **Autonomous Processing**: Fully autonomous enterprise module generation with minimal human intervention

### 3. **📚 Knowledge Management**
- Create comprehensive enterprise documentation
- Maintain enterprise infrastructure knowledge bases
- Standardize enterprise infrastructure patterns
- Enable enterprise knowledge sharing across teams
- **Research-Driven**: Deep research and analysis for enterprise-grade solutions

### 4. **🛡️ Security & Compliance**
- Ensure enterprise security best practices
- Validate enterprise compliance requirements
- Automate enterprise security scanning
- Maintain enterprise audit trails
- **Enterprise-Grade Security**: Comprehensive security analysis and validation for enterprise environments

## 🚀 Getting Started

### Prerequisites
- Python 3.12+
- Terraform CLI
- AWS CLI (for deployment)
- Required Python packages (see pyproject.toml)
- **⏱️ Time Allocation**: Allow 20-25 minutes per enterprise module generation

### Installation

#### Option 1: Standalone Installation

1. **Install [uv](https://docs.astral.sh/uv/getting-started/installation/)** for dependency management
2. **Create and activate a virtual environment with Python 3.12:**
   ```sh
   uv venv --python=3.12
   source .venv/bin/activate  # On Unix/macOS
   # or
   .venv\Scripts\activate  # On Windows
   ```
3. **Install dependencies from pyproject.toml:**
   ```sh
   uv pip install -e .
   ```
4. **Create a `.env` file and add the following environment variables:**
   ```sh
   OPENAI_API_KEY=XXXXXXXXX
   ```
   > **Note:** All available configuration options can be found in [`aws_orchestrator_agent/config/default.py`](aws_orchestrator_agent/config/default.py). You can set any of these options via your `.env` file to customize the AWS Orchestrator Agent's behavior.

5. **Start the A2A server with the agent card:**
   ```sh
   uv run --active aws-orchestrator-agent \
     --host localhost \
     --port 10102 \
     --agent-card aws_orchestrator_agent/card/aws_orchestrator_agent.json
   ```

#### Option 2: Docker Hub (Recommended for Quick Start)

**Pull and run the pre-built image from Docker Hub:**

```bash
# Pull the latest version
docker pull sandeep2014/aws-orchestrator-agent:latest

# Run the A2A server
docker run -d -p 10102:10102 \
  -e OPENAI_API_KEY=your_openai_key_here \
  --name aws-orchestrator \
  sandeep2014/aws-orchestrator-agent:latest
```

**Available Docker tags:**
- `latest` - Latest stable release
- `v0.1.0` - Specific version

#### Option 3: Build Docker Image Locally

**Build from source:**
```bash
# Clone the repository
git clone https://github.com/talkops-ai/aws-orchestrator-agent.git
cd aws-orchestrator-agent

# Build Docker image
docker build -t aws-orchestrator-agent:local .

# Run as A2A agent server
docker run -d -p 10102:10102 \
  -e OPENAI_API_KEY=your_key \
  -e ANTHROPIC_API_KEY=your_key \
  --name aws-orchestrator \
  aws-orchestrator-agent:local
```

---

### A2A Protocol Server

This agent runs on the Google A2A protocol to communicate with other A2A agents. To enable this, you will need an agent card so that other clients or servers can interact with this service.

**Alternative server startup with custom host and port:**
```sh
uv run aws-orchestrator-agent \
  --agent-card aws_orchestrator_agent/card/aws_orchestrator_agent.json \
  --host 0.0.0.0 \
  --port 10102
```
- `--host`: The host address to bind the server (default: `localhost`)
- `--port`: The port to run the server on (default: `10102`)

---

## 🧑‍💻 Interacting with the Orchestrator Agent

> **⚠️ Important**: Regardless of how you installed the agent (Standalone, Docker Hub, or Local Build), you need the Python client to interact with the server.

### Installing the Client

The Python client is included in the repository. Clone it to get the client:

```bash
# Clone the repository (if you haven't already)
git clone https://github.com/talkops-ai/aws-orchestrator-agent.git
cd aws-orchestrator-agent

# Install client dependencies (optional, only if you get import errors)
pip install httpx colorama
```

### Using the Client

**Start the client:**
```sh
python aws_orchestrator_client/client.py
```

> **Note:** You can override the following options when running the client:
> - `--agent`: The URL of the Orchestrator Agent server to connect to (default: `http://localhost:10102`)
> - `--session`: The session ID to use (default: `0`)
> - `--history`: Whether to show session history (default: `False`)
>
> **Example:**
> ```sh
> python aws_orchestrator_client/client.py --agent http://localhost:10102 --session 1 --history True
> ```

### Example Workflow (Docker + Client)

```bash
# 1. Start the Docker container
docker run -d -p 10102:10102 \
  -e OPENAI_API_KEY=your_key \
  --name aws-orchestrator \
  sandeep2014/aws-orchestrator-agent:latest

# 2. Verify the server is running
docker logs aws-orchestrator

# 3. Clone the repository for the client
git clone https://github.com/talkops-ai/aws-orchestrator-agent.git
cd aws-orchestrator-agent

# 4. Run the client to interact with the server
python aws_orchestrator_client/client.py

# 5. In the client, request a Terraform module
# Example: "Create an S3 bucket module with versioning and encryption"
```

## 📚 Documentation

### Comprehensive Documentation
- **[Planner Sub-Supervisor Documentation](docs/PLANNER_SUB_SUPERVISOR_DOCUMENTATION.md)**: Detailed planner agent architecture
- **[Generator Swarm Documentation](docs/GENERATOR_SWARM_AGENT_DOCUMENTATION.md)**: Complete generator swarm analysis
- **[Writer React Agent Documentation](docs/WRITER_REACT_AGENT_DOCUMENTATION.md)**: File writing agent details
- **[Supervisor Agent Architecture](docs/supervisor-agent-architecture.md)**: Main supervisor architecture
- **[Agent Architecture Overview](docs/agent-architecture.md)**: General agent architecture


## 💬 Community & Support

Join our community to get help, share ideas, and contribute to the project!

### Discord Community

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20Our%20Community-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/2V8AAufgp6)

Connect with other users, developers, and contributors:
- 💡 **Get Help**: Ask questions and get support from the community
- 🚀 **Share Projects**: Showcase your Terraform modules and use cases
- 🐛 **Report Issues**: Discuss bugs and feature requests
- 🤝 **Collaborate**: Find contributors and collaborate on improvements
- 📢 **Stay Updated**: Get notified about new releases and updates

**[Join the Discord Server →](https://discord.gg/2V8AAufgp6)**

### Getting Support

- **Discord**: For real-time help and discussions, join our [Discord community](https://discord.gg/2V8AAufgp6)
- **GitHub Issues**: For bug reports and feature requests, use [GitHub Issues](https://github.com/yourusername/aws-orchestrator-agent/issues)
- **Documentation**: Check our comprehensive [documentation](#-documentation) for guides and references
- **Examples**: Explore example modules in the `modules/` directory

---

## 🤝 Contributing

Contributions are always welcome! Please follow the workflow below:

### Development Workflow
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Contributing Guidelines
- Follow the existing code style and conventions
- Add tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR
- Write clear, descriptive commit messages


## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

We would like to thank the following projects and organizations that made this system possible:

### Frameworks & Orchestration
- **[LangGraph](https://github.com/langchain-ai/langgraph)**: For the powerful multi-agent orchestration framework that enables sophisticated agent coordination
- **[LangChain](https://github.com/langchain-ai/langchain)**: For the comprehensive LLM integration tools and abstractions
- **[Google A2A Protocol](https://github.com/google/a2a)**: For the Agent-to-Agent communication protocol enabling enterprise agent ecosystems

### AI & LLM Providers
- **[Anthropic](https://www.anthropic.com/)**: For Claude models with advanced reasoning capabilities
- **[OpenAI](https://openai.com/)**: For GPT models and comprehensive AI API
- **[Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service)**: For enterprise-grade OpenAI model hosting
- **Additional Providers**: Support for multiple LLM providers ensuring flexibility and resilience

### Infrastructure & Tooling
- **[AWS](https://aws.amazon.com/)**: For the robust cloud infrastructure platform and comprehensive service ecosystem
- **[Terraform](https://www.terraform.io/)**: For the industry-standard infrastructure as code capabilities
- **[Python](https://www.python.org/)**: For the powerful, flexible programming language
- **[uv](https://github.com/astral-sh/uv)**: For fast, reliable Python package and project management

### Community
- **Open Source Community**: For the countless libraries, tools, and best practices that enable modern software development
- **DevOps & Infrastructure Community**: For continuous feedback, contributions, and support

---

**Built with ❤️ for the DevOps and Infrastructure community**
