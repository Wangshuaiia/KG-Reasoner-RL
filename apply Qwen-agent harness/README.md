# KG-Reasoner-RL · Apply Qwen-Agent Harness

This directory contains the **Qwen-Agent harness and closed-book baseline** for [KG-Reasoner-RL](../README.md). Model training is handled by [`../kg-reasoner/`](../kg-reasoner/README.md).

| Directory | Purpose |
| --- | --- |
| [`qwen-agent-kg/`](qwen-agent-kg/README.md) | Qwen-Agent `Assistant`, the `kg_search` tool, a real-KG integration interface, and KBQA evaluation |
| [`qwen2.5-7b-4bit/`](qwen2.5-7b-4bit/README.md) | Qwen2.5-7B 4-bit on Apple Silicon through MLX, with chat and a closed-book baseline |

Both evaluation scripts locate `../../kg-reasoner/` relative to their own files and reuse its dataset loading and answer matching.

## Environment

```bash
# Run from this directory; recreate the environment after moving machines.
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

A real KG is not required to inspect the interface or run the closed-book baseline. No graph data, KG implementation, or default graph endpoint is bundled with this harness.

## Closed-book baseline

```bash
cd qwen-agent-kg
../.venv/bin/python ../qwen2.5-7b-4bit/kbqa_no_kg.py \
  --datasets cwq WebQSP WebQuestions graliqa --n 20 \
  --out_dir results/no_kg_current_paper
```

## Real-KG integration

The integration boundary is `qwen-agent-kg/kg_interface.py`. Implement its abstract `KnowledgeGraph.lookup(entity, question, relation, k)` method against an existing KG, then supply the implementation as `--kg_backend MODULE:CLASS`. An optional `--kg_config` JSON file passes constructor keyword arguments to that class. See the [interface contract](qwen-agent-kg/README.md#real-kg-interface).

Only the interface and backend-loading mechanism are provided. KG-assisted evaluation fails before model execution or result creation when the backend is missing or invalid. There is no automatic graph fallback. The training module's existing SPARQL/remote implementations remain available separately; adapting their entity resolution, ranking, and backtracking to this interface is future integration work.

Use `../.venv/bin/python run_agent_kbqa.py --help` from `qwen-agent-kg/` to inspect the CLI without a graph or model server. After a backend is implemented, the harness uses Qwen native tool calls through an OpenAI-compatible model service; `--model` and `--server` configure that service.

Use the same `--n` and `--seed` for paired comparisons with the closed-book baseline. Explicitly pass `--datasets cwq WebQSP WebQuestions graliqa` for the current manuscript's four benchmarks. Training checkpoint deployment and protocol adaptation remain unfinished.

The retained results are closed-book baseline records. No KG-assisted results are supplied for the new interface.
