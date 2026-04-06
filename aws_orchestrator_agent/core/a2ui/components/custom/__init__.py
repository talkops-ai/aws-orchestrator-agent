"""
Custom A2UI Components for the AWS Orchestrator Agent.

These components are domain-specific and require the AWS Orchestrator
custom catalog. Import this package to register them.
"""

from aws_orchestrator_agent.core.a2ui.components.custom.generated_module_review import (  # noqa: F401
    GeneratedModuleReviewComponent,
)
