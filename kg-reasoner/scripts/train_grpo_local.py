"""Single-process GRPO (Sec. 4.3, Eq. 9) for small models and single-device runs.

Same rollout (Algorithm 1), retrieval-token masking and rewards (Eq. 5-8) as the OpenRLHF
trainer (scripts/grpo_train.sh), without Ray/vLLM/DeepSpeed:
  * G rollouts per question, group-normalized advantages A = (R - mean) / std
  * clipped surrogate (clip 0.2) + k3 KL penalty to the reference model (coef 1e-5)
  * epoch 1 on the full set, later epochs on the hard-case set (hard-case resampling)

    python scripts/train_grpo_local.py --model ckpt/sft --train_file data/rl/train.jsonl \
        --hard_case_file data/rl/hard.jsonl --kg_backend remote --kg_server_url http://127.0.0.1:5003
"""
import argparse
import copy
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from kg_reasoner.data import load_records  # noqa: E402
from kg_reasoner.kg import add_kg_args, tool_factory_from_args  # noqa: E402
from kg_reasoner.rewards import build_judge, compute_reward  # noqa: E402
from kg_reasoner.rollout import HFGenerator, RolloutConfig, make_episode, run_episodes  # noqa: E402
from kg_reasoner.training import pick_device, token_logprobs  # noqa: E402


def episode_loss(policy, ref, ep, adv, args, device):
    seq = torch.tensor([ep.full_ids], device=device)
    n_prompt = len(ep.prompt_ids)
    logp = token_logprobs(policy, seq)[0, n_prompt - 1:]
    with torch.no_grad():
        ref_logp = token_logprobs(ref, seq)[0, n_prompt - 1:]
    old_logp = logp.detach()  # one policy update per rollout batch (mu = 1)
    mask = torch.tensor(ep.response_mask, device=device, dtype=torch.float)  # retrieved tokens masked out
    ratio = torch.exp(logp - old_logp)
    surr = torch.min(ratio * adv, torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv)
    log_r = ref_logp - logp
    kl = torch.exp(log_r) - log_r - 1  # k3 estimator
    loss = (-(surr) + args.kl_coef * kl) * mask
    return loss.sum() / mask.sum().clamp(min=1), (kl * mask).sum() / mask.sum().clamp(min=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="SFT warm-up checkpoint")
    p.add_argument("--train_file", required=True)
    p.add_argument("--hard_case_file", default=None, help="used from epoch 2 on (hard-case resampling)")
    p.add_argument("--out", default="ckpt/grpo")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=16, help="questions per step")
    p.add_argument("--n_samples", type=int, default=16, help="rollouts per question (G)")
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--kl_coef", type=float, default=1e-5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max_turns", type=int, default=8)
    p.add_argument("--max_tokens_per_turn", type=int, default=512)
    p.add_argument("--max_response_tokens", type=int, default=4096)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--judge", default="rule", choices=["rule", "hf"])
    p.add_argument("--judge_model", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_file", default=None)
    add_kg_args(p)
    args = p.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    policy = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    ref = copy.deepcopy(policy).eval()
    for prm in ref.parameters():
        prm.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    tools = tool_factory_from_args(args)
    judge = build_judge(args.judge, args.judge_model)
    gen = HFGenerator(policy, tok, temperature=args.temperature)
    cfg = RolloutConfig(max_turns=args.max_turns, max_tokens_per_turn=args.max_tokens_per_turn,
                        max_response_tokens=args.max_response_tokens)
    full = load_records(args.train_file)
    hard = load_records(args.hard_case_file) if args.hard_case_file else full
    log = open(args.log_file, "a") if args.log_file else None

    step = 0
    for epoch in range(args.epochs):
        data = list(full if epoch == 0 else hard)
        random.shuffle(data)
        print(f"== epoch {epoch + 1}: {len(data)} questions ({'full set' if epoch == 0 else 'hard cases'})")
        for b in range(0, len(data), args.batch_size):
            t0 = time.time()
            batch = data[b:b + args.batch_size]
            policy.eval()
            eps = [make_episode(r, tok, tools) for r in batch for _ in range(args.n_samples)]
            with torch.no_grad():
                run_episodes(eps, gen, tok, cfg)
            rewards = [compute_reward(ep.response_text, ep.record["answers"], ep.record["question"], judge) for ep in eps]
            r = torch.tensor([x["reward"] for x in rewards]).view(len(batch), args.n_samples)
            adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-6)).flatten()  # GRPO group norm

            policy.train()
            opt.zero_grad()
            kls = []
            for ep, a in zip(eps, adv.tolist()):
                loss, kl = episode_loss(policy, ref, ep, a, args, device)
                (loss / len(eps)).backward()
                kls.append(kl.item())
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            step += 1
            stats = {"step": step, "epoch": epoch + 1, "reward": r.mean().item(),
                     "r_answer": sum(x["r_answer"] for x in rewards) / len(rewards),
                     "r_format": sum(x["r_format"] for x in rewards) / len(rewards),
                     "n_search": sum(ep.n_search for ep in eps) / len(eps),
                     "kl": sum(kls) / len(kls), "secs": round(time.time() - t0, 1)}
            print(json.dumps(stats))
            if log:
                log.write(json.dumps({**stats, "sample": eps[0].response_text[:2000]}) + "\n")
                log.flush()
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break
    os.makedirs(args.out, exist_ok=True)
    policy.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
