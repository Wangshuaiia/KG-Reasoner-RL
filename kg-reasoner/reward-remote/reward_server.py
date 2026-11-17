"""Remote reward server for OpenRLHF (paper Sec. 4.3, Eq. 5-8).

    python reward-remote/reward_server.py --data_path data/rl/train.jsonl --port 1278 \
        --answer_judge hf --judge_model meta-llama/Llama-3.2-3B-Instruct

POST /get_reward {"query": [decoded prompt+response, ...]} -> {"rewards": [...]}
R_final = R_search + R_format + R_answer. The gold answers are looked up by the question text
parsed from the prompt (the prompt format is fixed by kg_reasoner.prompts.build_prompt).
"""
import argparse
import glob
import json
import logging
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from kg_reasoner.data import load_records  # noqa: E402
from kg_reasoner.prompts import split_query  # noqa: E402
from kg_reasoner.rewards import build_judge, compute_reward, normalize_answer  # noqa: E402

logger = logging.getLogger("reward_server")
DEFAULT_SPECIAL = ["<|endoftext|>", "<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|begin_of_text|>", "</s>", "<s>",
                   "<|end▁of▁sentence|>", "<|begin▁of▁sentence|>"]


class KGReasonerReward:
    def __init__(self, data_paths, judge, special_tokens, reasoning_judge=None, reasoning_weight=0.0, log_file=None):
        self.answers = {}
        for path in data_paths:
            for r in load_records(path):
                self.answers[normalize_answer(r["question"])] = (r["question"], r["answers"])
        logger.info("loaded %d questions", len(self.answers))
        self.judge, self.reasoning_judge, self.reasoning_weight = judge, reasoning_judge, reasoning_weight
        self.special = re.compile("|".join(re.escape(t) for t in sorted(set(special_tokens), key=len, reverse=True)))
        self.log_file = log_file

    def __call__(self, queries):
        rewards = []
        for q in queries:
            question, _, response = split_query(self.special.sub("", q))
            hit = self.answers.get(normalize_answer(question or ""))
            if hit is None:
                logger.warning("question not found in the training data: %r", (question or q)[:200])
                rewards.append(0.0)
                continue
            out = compute_reward(response, hit[1], hit[0], self.judge, self.reasoning_judge, self.reasoning_weight)
            rewards.append(out["reward"])
            if self.log_file:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"question": hit[0], "answers": hit[1], "response": response, **out},
                                       ensure_ascii=False) + "\n")
        return rewards


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", nargs="+", required=True, help="RL training file(s) (also accepts globs)")
    p.add_argument("--answer_judge", default="rule", choices=["rule", "hf"])
    p.add_argument("--judge_model", default="meta-llama/Llama-3.2-3B-Instruct", help="f_a in the paper")
    p.add_argument("--reasoning_judge_model", default=None, help="optional f_r (Llama-3.3-70B in the paper)")
    p.add_argument("--reasoning_weight", type=float, default=0.0, help="0 = Eq. 8 exactly")
    p.add_argument("--tokenizer", default=None, help="strip this tokenizer's special tokens from queries")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=1278)
    p.add_argument("--log_file", default=None)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    paths = [f for pat in args.data_path for f in (glob.glob(pat) or [pat])]
    special = list(DEFAULT_SPECIAL)
    if args.tokenizer:
        from transformers import AutoTokenizer

        special += AutoTokenizer.from_pretrained(args.tokenizer).all_special_tokens
    judge = build_judge(args.answer_judge, args.judge_model)
    rj = build_judge("hf", args.reasoning_judge_model) if args.reasoning_judge_model and args.reasoning_weight > 0 else None
    reward = KGReasonerReward(paths, judge, special, rj, args.reasoning_weight, args.log_file)

    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    app = FastAPI()

    @app.post("/get_reward")
    async def get_reward(request: Request):
        data = await request.json()
        return JSONResponse({"rewards": reward(data.get("query", []))})

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
