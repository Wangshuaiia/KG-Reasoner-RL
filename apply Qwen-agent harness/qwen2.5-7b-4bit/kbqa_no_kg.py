"""Closed-book baseline: Qwen2.5-7B-Instruct (4-bit, MLX) answers KBQA questions without a KG.

Samples questions from the 8 KG-Reasoner test sets, asks the model directly (greedy decoding,
answer only) and scores Hits@1 with KG-Reasoner's answer matching. The optional paper
reference scores below are retained from the earlier experiment, not the current manuscript.

    python kbqa_no_kg.py --n 20                     # 20 questions per dataset
    python kbqa_no_kg.py --datasets qald_10-en creak --n 50
"""
import argparse
import json
import os
import random
import sys
import time

from mlx_lm import load, stream_generate

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "kg-reasoner"))
sys.path.insert(0, REPO)
from kg_reasoner.data import load_records  # noqa: E402
from kg_reasoner.rewards import answer_match  # noqa: E402

MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
DATASETS = ["cwq", "WebQSP", "WebQuestions", "graliqa", "qald_10-en", "T-REX", "Zero_Shot_RE", "creak"]
# Legacy paper reference, "LLM w/o KG", Qwen-2.5-7B (Wikidata datasets only).
PAPER_QWEN7B_NO_KG = {"qald": 41.88, "trex": 31.15, "zeroshotre": 7.84, "creak": 73.26}

PROMPTS = {
    "qa": "Answer the question. Reply with only the single best answer entity, without explanation.\n"
          "Question: {q}",
    "slot_filling": "The input has the form 'subject [SEP] relation'. Give the object that completes the fact. "
                    "Reply with only the answer entity, without explanation.\nInput: {q}",
    "fact_checking": "Is the following claim true or false? Reply with only 'true' or 'false'.\nClaim: {q}",
}


def ask(model, tokenizer, record, max_tokens):
    msgs = [{"role": "user", "content": PROMPTS[record["task"]].format(q=record["question"])}]
    prompt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
    text, last = "", None
    for last in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens):  # greedy
        text += last.text
    return text.strip(), last


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL)
    p.add_argument("--datasets", nargs="+", default=DATASETS, help="file names in kg-reasoner/data/test_set")
    p.add_argument("--n", type=int, default=20, help="questions sampled per dataset")
    p.add_argument("--max_tokens", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    args = p.parse_args()

    t0 = time.time()
    model, tokenizer = load(args.model)
    print(f"model loaded in {time.time() - t0:.1f}s")
    os.makedirs(args.out_dir, exist_ok=True)
    rows, summary, peak = [], {}, 0.0
    for name in args.datasets:
        records = load_records(os.path.join(REPO, "data", "test_set", f"{name}.json"))
        sample = random.Random(args.seed).sample(records, min(args.n, len(records)))
        hits, t1 = 0, time.time()
        for rec in sample:
            pred, stats = ask(model, tokenizer, rec, args.max_tokens)
            hit = answer_match(pred, rec["answers"])
            hits += hit
            peak = max(peak, stats.peak_memory)
            rows.append({"dataset": rec["dataset"], "id": rec["id"], "question": rec["question"],
                         "answers": rec["answers"][:10], "pred": pred, "hit": hit})
        ds = sample[0]["dataset"] if sample else name
        summary[ds] = {"n": len(sample), "hits@1": round(100 * hits / max(1, len(sample)), 2),
                       "paper_qwen2.5_7b_no_kg": PAPER_QWEN7B_NO_KG.get(ds),
                       "sec_per_question": round((time.time() - t1) / max(1, len(sample)), 2)}
        print(f"{ds:12s} Hits@1 {summary[ds]['hits@1']:6.2f}  (n={len(sample)}, "
              f"{summary[ds]['sec_per_question']}s/q, paper w/o KG: {summary[ds]['paper_qwen2.5_7b_no_kg']})")

    with open(os.path.join(args.out_dir, "kbqa_no_kg.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, "kbqa_no_kg_summary.json"), "w") as f:
        json.dump({"model": args.model, "n_per_dataset": args.n, "seed": args.seed,
                   "peak_memory_gb": round(peak, 2), "total_minutes": round((time.time() - t0) / 60, 1),
                   "results": summary}, f, indent=2)
    print(f"peak memory {peak:.2f} GB, total {(time.time() - t0) / 60:.1f} min -> {args.out_dir}")


if __name__ == "__main__":
    main()
