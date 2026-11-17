"""SFT warm-up (Sec. 4.1 / 5.4 "Effect of SFT Warm-Up"): teach the model the tool-calling
format before RL. Tool-inserted <triples> blocks are excluded from the loss.

Single-device trainer (works on CPU / MPS / one GPU). For 8B/30B backbones use the OpenRLHF SFT
launcher in scripts/sft_train_openrlhf.sh instead.

    python scripts/sft_warmup.py --model Qwen/Qwen2.5-7B-Instruct --data data/sft/warmup.jsonl --out ckpt/sft
"""
import argparse
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from kg_reasoner.training import build_sft_example, pick_device, sft_loss  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="ckpt/sft")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_len", type=int, default=4096)
    p.add_argument("--max_steps", type=int, default=None, help="stop after this many optimizer steps")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    if device == "cuda":
        model.gradient_checkpointing_enable()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    examples = [build_sft_example(tok, r["prompt"], r["response"], args.max_len) for r in rows]
    print(f"{len(examples)} examples; supervised tokens: {sum(sum(l != -100 for l in y) for _, y in examples)}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    step, micro = 0, 0
    for epoch in range(args.epochs):
        random.shuffle(examples)
        for i in range(0, len(examples), args.batch_size):
            loss = sft_loss(model, examples[i:i + args.batch_size], pad, device) / args.grad_accum
            loss.backward()
            micro += 1
            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                step += 1
                print(f"epoch {epoch + 1} step {step}: loss {loss.item() * args.grad_accum:.4f}")
                if args.max_steps and step >= args.max_steps:
                    break
        if args.max_steps and step >= args.max_steps:
            break
    if micro % args.grad_accum:
        opt.step()
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
