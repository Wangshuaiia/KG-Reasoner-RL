"""Evaluate a trained KG-Reasoner model with Hits@1 (paper Sec. 5.1).

    python evaluation/eval_kg_reasoner.py --model_path ckpt/grpo --backend vllm \
        --data data/test_set/cwq.json data/test_set/WebQSP.json --kg_backend remote --kg_server_url http://127.0.0.1:5003
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from kg_reasoner.data import load_records  # noqa: E402
from kg_reasoner.kg import add_kg_args, tool_factory_from_args  # noqa: E402
from kg_reasoner.rewards import build_judge, check_format, extract_answer  # noqa: E402
from kg_reasoner.rollout import HFGenerator, RolloutConfig, VLLMGenerator, make_episode, run_episodes  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--exclude_ids", default=None, help="JSON list of ids to skip (e.g. used for training)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_turns", type=int, default=8)
    p.add_argument("--max_tokens_per_turn", type=int, default=512)
    p.add_argument("--max_response_tokens", type=int, default=4096)
    p.add_argument("--judge", default="rule", choices=["rule", "hf"], help="hf = Llama-3.2-3B judge as in the paper")
    p.add_argument("--judge_model", default=None)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--device", default="auto")
    p.add_argument("--output_dir", default="results")
    add_kg_args(p)
    args = p.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_path)
    if args.backend == "vllm":
        from vllm import LLM

        gen = VLLMGenerator(LLM(model=args.model_path, tensor_parallel_size=args.tensor_parallel_size,
                                gpu_memory_utilization=args.gpu_memory_utilization, trust_remote_code=True),
                            temperature=args.temperature)
    else:
        from transformers import AutoModelForCausalLM

        from kg_reasoner.training import pick_device

        model = AutoModelForCausalLM.from_pretrained(args.model_path).to(pick_device(args.device)).eval()
        gen = HFGenerator(model, tok, temperature=args.temperature)
    tools = tool_factory_from_args(args)
    judge = build_judge(args.judge, args.judge_model)
    cfg = RolloutConfig(max_turns=args.max_turns, max_tokens_per_turn=args.max_tokens_per_turn,
                        max_response_tokens=args.max_response_tokens)
    exclude = set(json.load(open(args.exclude_ids))) if args.exclude_ids else set()
    os.makedirs(args.output_dir, exist_ok=True)
    summary = {}
    for path in args.data:
        records = [r for r in load_records(path) if r["id"] not in exclude][: args.max_samples]
        name = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(args.output_dir, f"{name}-{os.path.basename(args.model_path.rstrip('/'))}.jsonl")
        stats = defaultdict(float)
        with open(out_path, "w", encoding="utf-8") as f:
            for b in range(0, len(records), args.batch_size):
                eps = [make_episode(r, tok, tools) for r in records[b:b + args.batch_size]]
                run_episodes(eps, gen, tok, cfg)
                for ep in eps:
                    pred = extract_answer(ep.response_text)
                    hit = judge(ep.record["question"], pred, ep.record["answers"]) if pred else 0.0
                    stats["hits@1"] += hit
                    stats["format_ok"] += float(check_format(ep.response_text))
                    stats["n_search"] += ep.n_search
                    f.write(json.dumps({"id": ep.record["id"], "question": ep.record["question"],
                                        "answers": ep.record["answers"], "pred": pred, "hit": hit,
                                        "n_search": ep.n_search, "finish": ep.finish_reason,
                                        "response": ep.response_text}, ensure_ascii=False) + "\n")
        n = max(1, len(records))
        summary[name] = {"n": len(records), **{k: round(v / n, 4) for k, v in stats.items()}}
        print(name, summary[name])
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
