# KG-Reasoner-RL · Qwen2.5-7B Closed-Book Baseline

This is the no-KG baseline for the [Qwen-Agent harness](../README.md). The current manuscript
uses four main datasets: CWQ, WebQSP, WebQuestions and GrailQA. The eight-dataset run and
legacy paper reference values below are retained as historical experiment records.

Model: [`mlx-community/Qwen2.5-7B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit), run with
[MLX](https://github.com/ml-explore/mlx) on the M2 GPU. The weights (~4.3 GB) are cached in `~/.cache/huggingface/hub`.
At runtime the model uses about 4.5 GB of memory and generates about 19 tokens/s.

## Setup
The virtual environment is shared by all experiments in `../`:
```bash
# From the kg-reasoner-rl project root
cd "apply Qwen-agent harness"
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Chat
```bash
cd qwen2.5-7b-4bit
../.venv/bin/python chat.py "What timezone is Utah in?"   # one question
../.venv/bin/python chat.py                              # interactive; empty line to quit
```

## Closed-book KBQA (no KG)
`kbqa_no_kg.py` samples questions from the 8 test sets in `../../kg-reasoner/data/test_set`, asks the model directly
with greedy decoding and an answer-only prompt, and scores Hits@1 with KG-Reasoner's answer matching.
The paper reference values below come from the earlier experiment context, not the current manuscript.
```bash
../.venv/bin/python kbqa_no_kg.py --n 20                          # 20 per dataset, about 2 min
../.venv/bin/python kbqa_no_kg.py --datasets qald_10-en creak --n 100
```
The script writes per-question outputs to `results/kbqa_no_kg.jsonl` and the summary to `results/kbqa_no_kg_summary.json`.

Run of 2026-11-12, 20 questions per dataset, seed 0:

| Dataset | Hits@1 | Legacy paper reference, Qwen-2.5-7B w/o KG |
|---|---|---|
| CWQ | 20.0 | – |
| WebQSP | 60.0 | – |
| WebQuestions | 45.0 | – |
| GrailQA | 5.0 | – |
| QALD10-en | 30.0 | 41.88 |
| T-REx | 60.0 | 31.15 |
| Zero-Shot RE | 5.0 | 7.84 |
| Creak | 80.0 | 73.26 |

With 20 questions per dataset the uncertainty is roughly ±10 points. The model here is 4-bit quantized, and the prompt
differs from the legacy paper setup, so these numbers are only a rough reference.
