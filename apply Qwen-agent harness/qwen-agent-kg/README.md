# KG-Reasoner-RL · Qwen-Agent KG Harness

This is the Qwen-Agent harness module of [KG-Reasoner-RL](../../README.md). Model training lives in [`../../kg-reasoner/`](../../kg-reasoner/README.md).

The harness provides a `kg_search(entity, relation?)` tool backed by an explicit real-KG interface. No graph data, concrete backend, or default endpoint is included. You can inspect the code and interface without a KG; KG-assisted evaluation requires a separately implemented backend.

## Files

| File | Role |
| --- | --- |
| `kg_interface.py` | Abstract real-KG lookup contract and loader for an external `MODULE:CLASS` backend |
| `kg_tool.py` | Qwen-Agent `kg_search` tool, question context, and lookup traces |
| `run_agent_kbqa.py` | KBQA evaluation using Qwen-Agent `Assistant`, Hits@1 scoring, and optional paired closed-book comparisons |
| `tests/test_kg_interface.py` | Offline interface, configuration, and tool-boundary checks |

## Real-KG interface

Implement a subclass of `kg_interface.KnowledgeGraph` in an importable Python module. The required method is:

```python
def lookup(
    self,
    entity: str,
    question: str = "",
    relation: str = "",
    k: int = 10,
) -> tuple[str, list[tuple[str, str, str]]]:
    ...
```

The return value is `(canonical_entity_name, triples)`, where every triple is `(head, relation, tail)`.

- Resolve the requested entity against the actual graph.
- Return at most `k` one-hop triples, each attached to the resolved entity as head or tail.
- Use `question` and `relation` to rank or filter existing graph evidence. Facts must come from the graph.
- Return `("", [])` for an unknown entity, or `(canonical_name, [])` when a known entity has no matching facts.
- Raise an exception for connection or query failures. An unavailable graph must not appear as a successful lookup with no facts.

There is no default implementation. The abstract class cannot be instantiated. The training module's `kg_reasoner.kg.base.KnowledgeGraph` separately exposes ID-based `neighbors` and `resolve` methods; a future adapter can bridge those methods to this name-based interface. No training-side KG/GNN/backtracking adapter is bundled here.

## Configuration and execution

Install the [harness dependencies](../README.md#environment). To inspect the CLI without a graph or running model:

```bash
../.venv/bin/python run_agent_kbqa.py --help
```

The CLI requires `--kg_backend MODULE:CLASS`. The class must inherit from `kg_interface.KnowledgeGraph` and implement `lookup`. Its module must be importable in the harness Python environment. `--kg_config PATH` optionally reads a JSON object and passes its entries as constructor keyword arguments; configuration is not copied into result metadata.

**After implementing your backend**, start the local model in a separate terminal from this directory:

```bash
../.venv/bin/python -m mlx_lm server \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit --port 8080
```

Then run the following integration example, replacing `my_kg:RealKG` and the configuration file with your implementation. These are placeholders, not provided modules or files:

```bash
../.venv/bin/python run_agent_kbqa.py \
  --kg_backend my_kg:RealKG --kg_config /path/to/kg-config.json \
  --datasets cwq WebQSP WebQuestions graliqa --n 20 \
  --no_kg_results results/no_kg_current_paper/kbqa_no_kg.jsonl \
  --tag qwen_agent_real_kg
```

Backend configuration is validated before model execution, dataset evaluation, or result creation. Omitting a backend is an error. There is no empty graph default or automatic fallback. Backend lookup failures propagate through the tool; evaluation records uncaught agent errors in its per-question `error` field.

The model service uses Qwen's native `<tool_call>` format through an OpenAI-compatible API. `use_raw_api: True` passes the tool schema directly. The default local service is MLX; the model API is separate from the KG interface.

Options include `--top_k` (default 10), `--server`, `--model`, `--n`, and `--seed`. `QWEN_AGENT_MAX_LLM_CALL_PER_RUN` defaults to 8. Use matching sample sizes and seeds for paired comparisons with [`../qwen2.5-7b-4bit/kbqa_no_kg.py`](../qwen2.5-7b-4bit/kbqa_no_kg.py).

## Outputs and validation

`results/<tag>.jsonl` contains predictions, searches, trajectories, and errors. `results/<tag>_summary.json` records the backend class identifier, model, evaluation settings, and metrics. No KG-assisted results are supplied for this interface; retained result files are closed-book baselines.

Run the offline interface tests from this directory:

```bash
../.venv/bin/python -m unittest discover -s tests -v
```

The tests use isolated test fixtures and require no graph server, model weights, or network calls.
