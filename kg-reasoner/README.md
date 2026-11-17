# KG-Reasoner-RL · Model Training

This directory is the **model training module** of [KG-Reasoner-RL](../README.md), accompanying the current manuscript, **KG-Reasoner: Reinforcement Learning for Agentic Knowledge Graph Reasoning**.

Manuscript source: [sample-sigconf-authordraft.tex](../../paper-kg-reasoner/sample-sigconf-authordraft.tex). The current version focuses on adapting Qwen2.5-7B to existing agent harnesses through RL. This document replaces the earlier paper title, arXiv reference, and eight-dataset main-experiment description.

This module handles data preparation, SFT warm-up, GRPO, hard multi-hop example selection, and training-side KG retrieval, GNN entity ranking, backtracking, and evaluation. The **Qwen-Agent harness lives in [`../apply Qwen-agent harness/`](<../apply Qwen-agent harness/README.md>)**, which handles agent execution and closed-book comparisons.

## Relationship between training and the harness

Training trajectories use the following format: the model emits `<search>entity</search>`, the KG tool returns `<triples>...</triples>`, and the model gives its answer in `<answer>...</answer>`. A backtracking request uses `<search>Backtrack=True</search>`.

The current harness uses Qwen's native `<tool_call>` format and `kg_search(entity, relation?)`, with `Final answer:` output. It reuses this module's data loading and answer matching, and exposes `kg_interface.KnowledgeGraph.lookup` for an external real-KG backend. No graph data or backend implementation is bundled with the harness, and evaluation requires an explicit `--kg_backend MODULE:CLASS`. Deploying training checkpoints, adapting the two tool protocols, and connecting the real KG/GNN/explicit backtracking tools to Qwen-Agent remain unfinished. Training rollout and evaluation loops here are implemented by `kg_reasoner`.

## Method-to-code mapping

| Component | Code |
| --- | --- |
| Tool-use prompts and multi-turn rollout | `kg_reasoner/prompts.py`, `kg_reasoner/rollout.py` |
| One-hop neighbor retrieval, including incoming and outgoing edges | `kg_reasoner/kg/sparql_kg.py` |
| GNN entity ranking | `kg_reasoner/kg/gnn.py`, `kg_reasoner/kg/rankers.py`, `scripts/train_gnn.py` |
| Retrieval paths and explicit backtracking | `kg_reasoner/kg/backtrack.py`, `kg_reasoner/kg/tool.py` |
| Warm-up trajectories and error-correction examples | `scripts/build_warmup_data.py` |
| SFT | `scripts/sft_warmup.py` (single device), `scripts/sft_train_openrlhf.sh` (DeepSpeed) |
| Search, format, and answer rewards | `kg_reasoner/rewards.py`, `reward-remote/reward_server.py` |
| GRPO | `scripts/grpo_train.sh`, `OpenRLHF-RAG/`; single-process alternative: `scripts/train_grpo_local.py` |
| Hard multi-hop example selection | `scripts/select_hard_cases.py` |
| Dataset loading and sampling | `kg_reasoner/data.py`, `scripts/build_rl_data.py` |
| Training-side Hits@1 evaluation | `evaluation/eval_kg_reasoner.py` |
| Qwen-Agent harness | `../apply Qwen-agent harness/qwen-agent-kg/` |

## Directory layout

```text
kg-reasoner/
├── kg_reasoner/                 Shared library for training, data, and evaluation
│   ├── prompts.py, data.py
│   ├── rollout.py, rewards.py, training.py
│   └── kg/                      KG, GNN, backtracking, and shared service
├── scripts/                     Data, GNN, SFT, GRPO, and hard-case selection
├── OpenRLHF-RAG/                OpenRLHF distributed training backend
├── reward-remote/reward_server.py
├── evaluation/eval_kg_reasoner.py
├── tests/
└── data/test_set/               Four main benchmarks and historical extra datasets
```

The distribution name is `kg-reasoner-rl-training`; the Python import name remains `kg_reasoner`.

## Environment

Run the following commands from this directory. Full multi-GPU training requires Linux / CUDA. The dependencies in `requirements.txt` include DeepSpeed, FlashAttention, Ray, and vLLM; install them in a separate training environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install -e OpenRLHF-RAG     # Required for multi-GPU training
python -m pytest -q tests
```

KG backends:

- `sparql`: a Virtuoso endpoint loaded with Freebase, configured through `--sparql_endpoint` or `SPARQL_ENDPOINT`; the default is `http://localhost:8890/sparql`.
- `remote`: a shared `kg_reasoner.kg.server` that maintains the KG connection, neighbor cache, and GNN ranker for concurrent rollouts.

The project does not include a complete Freebase installation, training splits, or model checkpoints. Prepare them separately. Other graphs can implement the `KnowledgeGraph` interface in `kg_reasoner/kg/base.py`.

## Training pipeline

The current manuscript uses Qwen2.5-7B, 500 SFT warm-up examples, 2,000 RL examples, and two RL epochs. The main datasets are CWQ, WebQSP, WebQuestions, and GrailQA. The full training setting uses eight A100 80GB GPUs and 16 rollouts per question.

### 1. Data and GNN

Place the four datasets' **training splits** in `data/training_set/`, using the corresponding test filenames: `cwq.json`, `WebQSP.json`, `WebQuestions.json`, and `graliqa.json`. The sampler reads every JSON/JSONL file in the source directory, so include only these four training splits for the current manuscript's setting.

```bash
python scripts/build_rl_data.py --source_dir data/training_set --n 2000 --out data/rl/train.jsonl
python scripts/train_gnn.py --train_file data/rl/train.jsonl --out ckpt/gnn.pt --sparql_endpoint http://localhost:8890/sparql
```

### 2. Shared KG service

Keep the service running in a separate terminal:

```bash
python -m kg_reasoner.kg.server --kg_backend sparql --sparql_endpoint http://localhost:8890/sparql --kg_ranker gnn --gnn_ckpt ckpt/gnn.pt --port 5003
```

### 3. Warm-up data and SFT

```bash
python scripts/build_warmup_data.py --train_file data/rl/train.jsonl --n 500 --out data/sft/warmup.jsonl --kg_backend remote --kg_server_url http://127.0.0.1:5003
BASE_MODEL=Qwen/Qwen2.5-7B-Instruct bash scripts/sft_train_openrlhf.sh
```

`build_warmup_data.py` defaults to `--teacher path`, which searches for KG paths to gold answers. It also supports `--teacher hf:MODEL` and `--teacher vllm:MODEL`. `--error_ratio` controls the proportion of trajectories with an injected error followed by backtracking and correction.

For single-device SFT, use the command below. It masks tool-returned `<triples>` tokens. The existing OpenRLHF SFT launcher trains on these tokens as well, so the two paths use different loss masking.

```bash
python scripts/sft_warmup.py --model Qwen/Qwen2.5-7B-Instruct --data data/sft/warmup.jsonl --out ckpt/sft
```

### 4. Hard cases and GRPO

Explicitly pass `--epsilon 1.0` to match the current manuscript. The script still defaults to the median loss, so its default does not reproduce the manuscript's threshold. Without a reference trajectory, the selector computes loss on a minimal gold-answer continuation. It also has a fallback for small datasets when no example passes the filter.

```bash
python scripts/select_hard_cases.py --model ckpt/sft --train_file data/rl/train.jsonl --reference_file data/sft/warmup.jsonl --epsilon 1.0 --out data/rl/hard.jsonl
```

Start the reward service in a separate terminal and keep it running:

```bash
python reward-remote/reward_server.py --data_path data/rl/train.jsonl --port 1278 --answer_judge hf --judge_model meta-llama/Llama-3.2-3B-Instruct
```

Then start Ray and GRPO:

```bash
bash scripts/ray_start.sh
SFT_MODEL=ckpt/sft bash scripts/grpo_train.sh
```

`grpo_train.sh` uses the full RL set in the first epoch and `data/rl/hard.jsonl` in the second. Generate the hard-case file as shown above to avoid the automatic selector's default median threshold. Regenerate the file after changing the input data or SFT checkpoint. The single-process alternative is `scripts/train_grpo_local.py`.

### 5. Training-side evaluation

```bash
python evaluation/eval_kg_reasoner.py --model_path ckpt/grpo/epoch2 --backend vllm --data data/test_set/cwq.json data/test_set/WebQSP.json data/test_set/WebQuestions.json data/test_set/graliqa.json --kg_backend remote --kg_server_url http://127.0.0.1:5003
```

This evaluates the training-side tag protocol. See the [harness README](<../apply Qwen-agent harness/README.md>) for Qwen-Agent execution.

## Implementation and reproduction notes

- The final reward is `R_search + R_format + R_answer`. Answers can be scored with rule-based matching or a model judge. The optional reasoning judge has weight 0 by default and is excluded from the default total reward.
- GRPO rollout masks tool-returned `<triples>` tokens from policy loss and KL. Single-device SFT also masks tool output; the OpenRLHF SFT path differs as described above.
- The current tool defaults to 5 triples per call and at most 8 searches per trajectory. The GNN uses a frozen MiniLM text encoder. These settings can be adjusted through arguments.
- QALD10-en, T-REx, Zero-Shot RE, and Creak remain supported for extra experiments. The dataset [README](data/test_set/README.md) records the historical inventory.
- Some inherited code comments still reference sections or appendices of the earlier paper. Use this README's functional mapping and the current manuscript to locate methods.
- Project organization did not retrain models or verify the current manuscript's tables. Training and the harness still differ in tool protocol and model precision; connecting the harness to a real KG backend remains an integration task.
