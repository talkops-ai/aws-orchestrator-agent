#!/bin/bash
set -e

# Use environment variables with defaults if not set
exec aws-orchestrator \
  --host "${A2A_HOST:-0.0.0.0}" \
  --port "${A2A_PORT:-10104}" \
  --agent-card "${A2A_AGENT_CARD:-aws_orchestrator_agent/card/aws_orchestrator_agent.json}" \
  "$@"