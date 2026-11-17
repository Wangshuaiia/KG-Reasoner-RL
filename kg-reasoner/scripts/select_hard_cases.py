"""Hard-case resampling (Sec. 4.3): after the first RL episode, train only on hard multi-hop samples.

Difficulty = next-token loss of the SFT warm-up model on the sample's reference trajectory (Eq. 10):
    L_sft = -1/T * sum_t log P(a_t | a_<t, s)
Samples with L_sft > epsilon that need multi-hop reasoning are kept.
Reference trajectory: the sample's warm-up trajectory if --reference_file has one for its id,
otherwise the minimal "</think> <answer>gold</answer>" continuation.
Multi-hop: dataset annotation (hops >= 2, e.g. CWQ, GrailQA num_edge, WebQSP inferential chain)
or a reference trajectory with >= 2 retrieval calls.

    python scripts/select_hard_cases.py --model ckpt/sft --train_file data/rl/train.jsonl --out data/rl/hard.jsonl
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from kg_reasoner.data import load_records, write_jsonl  # noqa: E402
from kg_reasoner.prompts import ANSWER_CLOSE, ANSWER_OPEN, THINK_CLOSE, build_prompt  # noqa: E402
from kg_reasoner.rewards import count_searches  # noqa: E402
from kg_reasoner.training import build_sft_example, pick_device, sequence_nll  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="SFT warm-up checkpoint")
    p.add_argument("--train_file", required=True)
    p.add_argument("--reference_file", default=None, help="JSONL with id/response (e.g. build_warmup_data.py output)")
    p.add_argument("--out", default="data/rl/hard.jsonl")
    p.add_argument("--epsilon", type=float, default=None, help="loss threshold; default: --quantile of the losses")
    p.add_argument("--quantile", type=float, default=0.5)
    p.add_argument("--no_multi_hop_filter", action="store_true")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = pick_device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    refs = {}
    if args.reference_file:
        for line in open(args.reference_file, encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                refs.setdefault(row["id"], row["response"])

    records = load_records(args.train_file)
    for r in records:
        resp = refs.get(r["id"]) or f" {THINK_CLOSE}\n{ANSWER_OPEN}{r['answers'][0]}{ANSWER_CLOSE}"
        ids, labels = build_sft_example(tok, build_prompt(r), resp)
        r["sft_loss"] = sequence_nll(model, ids, labels, device)
        r["multi_hop"] = bool((r.get("hops") or 0) >= 2 or count_searches(resp) >= 2)
    losses = sorted(r["sft_loss"] for r in records)
    eps = args.epsilon if args.epsilon is not None else losses[min(len(losses) - 1, int(args.quantile * len(losses)))]
    hard = [r for r in records if r["sft_loss"] > eps and (args.no_multi_hop_filter or r["multi_hop"])]
    if not hard:  # keep training possible on very small training sets
        print("[warning] no sample passed the filter; keeping the hardest samples regardless of hop count")
        hard = sorted(records, key=lambda r: -r["sft_loss"])[: max(1, len(records) // 2)]
    write_jsonl(args.out, hard)
    print(f"epsilon={eps:.4f}: {len(hard)}/{len(records)} hard multi-hop samples -> {args.out}")


if __name__ == "__main__":
    main()
