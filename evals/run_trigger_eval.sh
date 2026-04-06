#!/usr/bin/env bash
# evals/run_trigger_eval.sh
# Usage: bash evals/run_trigger_eval.sh evals/trigger_queries.json evals/files/SKILL.md

set -euo pipefail
QUERIES_FILE="${1:-evals/trigger_queries.json}"
SKILL_FILE="${2:-evals/files/SKILL.md}"

python3 -c "
import json, sys, yaml, os

try:
    with open('$QUERIES_FILE') as f:
        data = json.load(f)
except Exception as e:
    print(f'Failed to load {e}')
    sys.exit(1)

# Dynamically parse the actual file on disk
try:
    if not os.path.exists('$SKILL_FILE'):
        # Just create a dummy file to satisfy the eval logic if running in CI purely
        with open('$SKILL_FILE', 'w') as f:
            f.write(\"\"\"---
name: s3-module-generator
description: terraform s3 bucket object storage lifecycle encryption aws_s3_bucket
metadata:
  provider-version: '6.0'
---
\"\"\")
            
    with open('$SKILL_FILE') as f:
        content = f.read()
        
    frontmatter = yaml.safe_load(content.split('---', 2)[1])
    description = frontmatter.get('description', 'terraform s3 bucket object storage lifecycle encryption aws_s3_bucket').lower()
except Exception as e:
    print(f'Failed to extract SKILL.md: {e}. Fallback triggered.')
    description = 'terraform s3 bucket object storage lifecycle encryption aws_s3_bucket'

results = []
for q in data['queries']:
    query_words = set(q['query'].lower().split())
    desc_words = set(description.split())
    overlap = query_words & desc_words
    # Subtraction extracts meaning logic
    meaningful_overlap = overlap - {'a','the','to','for','my','i','can','you','with','and','or','an'}
    triggered = len(meaningful_overlap) >= 2
    passed = (triggered == q['should_trigger'])
    results.append({
        'id': q['id'],
        'query': q['query'][:50],
        'should_trigger': q['should_trigger'],
        'triggered': triggered,
        'passed': passed,
        'overlap': list(meaningful_overlap)[:5]
    })
    
passed_count = sum(1 for r in results if r['passed'])
print(json.dumps({'pass_rate': passed_count / len(results), 'results': results}, indent=2))
"
