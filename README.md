# KG-Reasoner-RL

**Reinforcement Learning for Agentic Knowledge Graph Reasoning**

This project accompanies the current manuscript, **KG-Reasoner: Reinforcement Learning for Agentic Knowledge Graph Reasoning**, and brings model training and Qwen-Agent harness experiments into one repository.

Manuscript source: [sample-sigconf-authordraft.tex](../paper-kg-reasoner/sample-sigconf-authordraft.tex). This link points to the sibling paper directory in the workspace. The manuscript is maintained separately and must be retained separately when moving this project.

## Module responsibilities

| Module | Responsibility | Documentation |
| --- | --- | --- |
| `kg-reasoner/` | **Model training:** training-data construction, GNN entity ranking, SFT warm-up, GRPO reinforcement learning, hard-case selection, training-side KG tools, and evaluation | [Training README](kg-reasoner/README.md) |
| `apply Qwen-agent harness/` | **Qwen-Agent harness:** multi-step KBQA using the existing `Assistant` and `kg_search` tool, with comparison against a closed-book baseline | [Harness README](<apply Qwen-agent harness/README.md>) |

The manuscript's workflow uses SFT and RL to train KG tool use and reasoning, then runs multi-hop question answering in an existing agent harness. Training trajectories use `<search>` / `<triples>` / `<answer>` tags. The current harness uses Qwen's native `<tool_call>` format, Qwen-Agent tool orchestration, and `Final answer:` output.

**Current integration status:** Both modules belong to one project. The harness loads datasets and answer-matching logic from this project's `kg-reasoner/` module. Training implements SPARQL/remote KG access, GNN ranking, and explicit backtracking. The harness exposes a real-KG `KnowledgeGraph.lookup` interface and requires an external implementation before evaluation; no graph data, backend implementation, or default endpoint is bundled with the harness. Its default model is MLX 4-bit Qwen2.5-7B. Deploying training checkpoints, adapting the tool protocols, and connecting training-side KG/GNN/backtracking components to the harness remain unfinished.

## Directory layout

```text
kg-reasoner-rl/
├── README.md
├── kg-reasoner-rl.code-workspace  # IDE workspace entry point
├── kg-reasoner/                  # Model training, copied from KG-Reasoner
│   ├── kg_reasoner/              # Python library; import name remains kg_reasoner
│   ├── scripts/                  # Data, GNN, SFT, GRPO, and hard-case selection
│   ├── OpenRLHF-RAG/             # Distributed training backend
│   ├── reward-remote/            # Reward service
│   ├── evaluation/               # Training-side evaluation
│   ├── tests/
│   └── data/test_set/
└── apply Qwen-agent harness/     # Qwen-Agent harness and baselines
    ├── requirements.txt
    ├── qwen-agent-kg/            # Assistant, kg_search, real-KG interface, traces
    ├── qwen2.5-7b-4bit/          # MLX model chat and closed-book baseline
    └── .venv/                   # Preserved local environment; ignored by Git
```

The project display name is **KG-Reasoner-RL**, the training distribution is named `kg-reasoner-rl-training`, and the Python import name remains `kg_reasoner`. The original source directories outside this project remain unchanged. The original training repository's `.git` history was not copied into a nested repository; this project has a single root Git repository. Closed-book baseline results are retained, and generated results and logs are ignored by Git by default.

## Alignment with the current manuscript

- Backbone: Qwen2.5-7B.
- Main benchmarks: CWQ, WebQSP, WebQuestions, and GrailQA, evaluated with Hits@1.
- Training design: 500 SFT warm-up examples, 2,000 RL examples, and two RL epochs; the second epoch selects difficult multi-hop questions. The manuscript sets the hard-case threshold to `epsilon = 1.0`.
- Training components: search, format, and answer rewards, plus GNN entity ranking and explicit backtracking.
- `data/test_set/` also retains four additional datasets supported by the earlier project. They are optional compatibility datasets, outside the current manuscript's four main benchmarks. The existing GrailQA filename is `graliqa.json`.

These are the manuscript settings. Some inherited script defaults and tool interfaces still require explicit configuration; see the module READMEs. Project organization did not retrain models or verify the manuscript's reported metrics.

## Getting started

Open `kg-reasoner-rl.code-workspace` in your IDE, or open this directory directly. The setup commands below start at the project root.

### Model training

Full distributed training uses Linux / CUDA and depends on DeepSpeed, FlashAttention, Ray, and vLLM. Use a separate environment from the local MLX harness.

```bash
cd kg-reasoner
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install -e OpenRLHF-RAG
```

Follow the [training pipeline](kg-reasoner/README.md#training-pipeline) to prepare the real KG, training splits, and checkpoints, then run SFT and GRPO.

### Qwen-Agent harness on Apple Silicon / MLX

The local environment is preserved at `apply Qwen-agent harness/.venv`, with its launcher paths updated. Recreate the environment after moving to another machine or changing the Python installation:

```bash
cd "apply Qwen-agent harness"
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

In terminal 1, start the model service from `apply Qwen-agent harness/`:

```bash
.venv/bin/python -m mlx_lm server \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit --port 8080
```

The KG interface can be inspected without a running model or graph. From `apply Qwen-agent harness/qwen-agent-kg/`, run `../.venv/bin/python run_agent_kbqa.py --help`. KG-assisted evaluation is unavailable until you implement and configure a real-KG backend; it fails during argument validation if `--kg_backend` is missing.

The closed-book baseline needs no KG. Run it from the same directory:

```bash
../.venv/bin/python ../qwen2.5-7b-4bit/kbqa_no_kg.py \
  --datasets cwq WebQSP WebQuestions graliqa --n 20 \
  --out_dir results/no_kg_current_paper
```

See the [KG interface contract](<apply Qwen-agent harness/qwen-agent-kg/README.md#real-kg-interface>) for the required lookup signature and external backend configuration. No KG-assisted evaluation has been run with this interface.
