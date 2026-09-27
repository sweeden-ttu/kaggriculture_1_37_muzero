# VSA agents (LangGraph + local Ollama)

| Agent | Role |
|---|---|
| [`kaggriculture_muzero_spec_agent.py`](kaggriculture_muzero_spec_agent.py) | Component architecture specs |
| [`kaggriculture_muzero_refactor_agent.py`](kaggriculture_muzero_refactor_agent.py) | Repo organize / gaps / IDE / conda / docs publish |
| [`local_llm.py`](local_llm.py) | Shared Ollama helper |

## Refactor agent

```bash
# List tasks
python docs/vsa/kaggriculture_muzero_refactor_agent.py --list

# Dry-run plan (no writes)
python docs/vsa/kaggriculture_muzero_refactor_agent.py all --offline
# or: make refactor-plan

# Apply safe scaffolding
python docs/vsa/kaggriculture_muzero_refactor_agent.py all --apply --offline
# or: make refactor-apply

# Single task + Ollama prioritization
python docs/vsa/kaggriculture_muzero_refactor_agent.py lsp --apply
python docs/vsa/kaggriculture_muzero_refactor_agent.py publish_docs --apply

# Destructive site cleanup / gh-deploy
python docs/vsa/kaggriculture_muzero_refactor_agent.py housekeeping --apply --force
python docs/vsa/kaggriculture_muzero_refactor_agent.py publish_docs --apply --force
```

Reports land in `docs/vsa/out/refactor_report.md`.
