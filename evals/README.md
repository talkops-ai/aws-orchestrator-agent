# Skill Quality Evaluation Framework

This repository uses automated, assertion-driven evaluation loops to test the quality of the generative instructions embedded within our Agent Skills. 
Unlike integration tests, these ensure the Agent actually understands the context correctly enough to build robust, secure Terraform templates.

## 1. Description Trigger Eval
Evaluates whether a skill's description will trigger when requested conceptually. It runs keyword intersections to assert should-trigger bounds.

```bash
bash evals/run_trigger_eval.sh evals/trigger_queries.json
```

## 2. Module Quality Eval
Evaluates actual generated module boundaries.
* **grade_module.py**: Uses Python and Terraform CLI assertions against the generated `outputs/` folder.

```bash
python3 evals/grade_module.py --eval-id 1 --output-dir path/to/outputs
```
