"""Reward functions (paper Sec. 4.3, Eq. 5-8) and answer matching for Hits@1.

    R_search = min(0.5 * n, 0.8)          n = number of retrieval actions        (Eq. 5)
    R_format = 0.5 if the tags are well formed else 0                              (Eq. 6)
    R_answer = 1 if the answer is correct else 0                                   (Eq. 7)
    R_final  = R_search + R_format + R_answer                                      (Eq. 8)

Answer correctness is judged by f_a (Llama-3.2-3B in the paper, `HFJudge`) or by rule-based
matching (`RuleJudge`, no model needed). Appendix A also mentions a reasoning scorer f_r
(Llama-3.3-70B); it is not part of Eq. 8, so it is optional (weight 0 by default).
"""
import re
import string
from typing import Dict, List, Optional

from .prompts import (ANSWER_CLOSE, ANSWER_OPEN, SEARCH_CLOSE, SEARCH_OPEN, THINK_CLOSE, THINK_OPEN, TRIPLES_CLOSE,
                      TRIPLES_OPEN)


# --------------------------------------------------------------------- parsing
def extract_answer(response: str) -> str:
    if ANSWER_OPEN not in response:
        return ""
    return response.rsplit(ANSWER_OPEN, 1)[-1].split(ANSWER_CLOSE, 1)[0].strip()


def count_searches(response: str) -> int:
    """Executed retrieval actions: </search> immediately followed by a <triples> block."""
    return len(re.findall(re.escape(SEARCH_CLOSE) + r"\s*" + re.escape(TRIPLES_OPEN), response))


def check_format(response: str) -> bool:
    """Response = text after "Assistant: <think>": reasoning (with search calls) </think> <answer>..</answer>."""
    body = response.strip()
    if body.count(THINK_OPEN) > 0 or body.count(THINK_CLOSE) != 1:
        return False
    if body.count(ANSWER_OPEN) != 1 or body.count(ANSWER_CLOSE) != 1:
        return False
    think, after = body.split(THINK_CLOSE)
    if SEARCH_OPEN in after or TRIPLES_OPEN in after:
        return False
    tail = after.split(ANSWER_CLOSE, 1)[1]
    if not after.strip().startswith(ANSWER_OPEN) or tail.strip():
        return False
    if not extract_answer(body):
        return False
    # every call is "<search> q </search>" followed by exactly one tool-inserted "<triples> .. </triples>",
    # with no stray or nested tags anywhere else in the reasoning
    tags = (SEARCH_OPEN, SEARCH_CLOSE, TRIPLES_OPEN, TRIPLES_CLOSE)
    pos = 0
    for m in _CALL_RE.finditer(think):
        if any(t in think[pos:m.start()] for t in tags) or any(t in m.group(1) for t in tags):
            return False
        if TRIPLES_OPEN in m.group(2) or SEARCH_OPEN in m.group(2):
            return False
        pos = m.end()
    return not any(t in think[pos:] for t in tags)


_CALL_RE = re.compile(re.escape(SEARCH_OPEN) + r"(.*?)" + re.escape(SEARCH_CLOSE) + r"\s*"
                      + re.escape(TRIPLES_OPEN) + r"(.*?)" + re.escape(TRIPLES_CLOSE), re.S)


# ------------------------------------------------------------------- matching
def normalize_answer(s: str) -> str:
    s = s.lower().replace("_", " ")
    s = "".join(ch if ch not in set(string.punctuation + "‘’´`") else " " for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


_BOOL = {"yes": "true", "true": "true", "correct": "true", "no": "false", "false": "false", "incorrect": "false"}


def answer_match(pred: str, golds: List[str], strict: bool = False) -> bool:
    """Hits@1 matching: normalized exact match or containment (as in ToG-style evaluation).
    strict=True: normalized exact match only (used to label KG entities, where containment
    would accept e.g. "United States-Mexico-Canada Agreement" for "Canada")."""
    p = normalize_answer(pred)
    if not p:
        return False
    for g in golds:
        g = normalize_answer(str(g))
        if not g:
            continue
        if g in _BOOL.values() or g in _BOOL:
            if _BOOL.get(p.split()[0] if p.split() else p, p) == _BOOL.get(g, g):
                return True
            continue
        if p == g:
            return True
        if not strict and (re.search(r"\b" + re.escape(g) + r"\b", p)
                           or (len(p) >= 4 and re.search(r"\b" + re.escape(p) + r"\b", g))):
            return True
    return False


class RuleJudge:
    def __call__(self, question: str, pred: str, golds: List[str]) -> float:
        return float(answer_match(pred, golds))


class HFJudge:
    """f_a: an instruction model decides whether the prediction matches the ground truth."""

    PROMPT = (
        "You are an expert in evaluating answer correctness for knowledge-based questions.\n"
        "Compare the predicted answer with the ground truth answers. If the prediction is correct, return '1'. "
        "If the prediction is incorrect, return '0'. Return only '1' or '0'.\n\n"
        "Question: {question}\nPredicted Answer: {pred}\nGround Truth: {gold}\nYour Answer:"
    )

    def __init__(self, model_path: str = "meta-llama/Llama-3.2-3B-Instruct", device: Optional[str] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype).to(self.device).eval()

    def _generate(self, prompt: str, max_new_tokens: int) -> str:
        import torch

        msgs = [{"role": "user", "content": prompt}]
        if getattr(self.tokenizer, "chat_template", None):
            text = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
        enc = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def __call__(self, question: str, pred: str, golds: List[str]) -> float:
        if not pred:
            return 0.0
        reply = self._generate(self.PROMPT.format(question=question, pred=pred, gold=" | ".join(golds)), 5)
        m = re.search(r"\b([01])\b", reply)
        return float(m.group(1)) if m else float(answer_match(pred, golds))

    def score_reasoning(self, question: str, reasoning: str) -> float:
        prompt = ("Evaluate whether the following multi-step knowledge-graph reasoning process is logically sound. "
                  "Reply with a single score between 0 and 1.\n\nQuestion: " + question
                  + "\nReasoning Process: " + reasoning[-6000:] + "\nScore:")
        m = re.search(r"(0(?:\.\d+)?|1(?:\.0+)?)", self._generate(prompt, 16))
        return float(m.group(1)) if m else 0.0


def build_judge(name: str = "rule", model_path: Optional[str] = None, device: Optional[str] = None):
    if name == "rule":
        return RuleJudge()
    if name == "hf":
        return HFJudge(model_path or "meta-llama/Llama-3.2-3B-Instruct", device)
    raise ValueError(f"Unknown judge {name!r} (rule | hf)")


# ---------------------------------------------------------------------- reward
def compute_reward(response: str, golds: List[str], question: str = "", judge=None,
                   reasoning_judge=None, reasoning_weight: float = 0.0) -> Dict[str, float]:
    judge = judge or RuleJudge()
    n = count_searches(response)
    r_search = min(0.5 * n, 0.8)
    r_format = 0.5 if check_format(response) else 0.0
    pred = extract_answer(response)
    r_answer = judge(question, pred, golds) if pred else 0.0
    total = r_search + r_format + r_answer
    out = {"reward": total, "r_search": r_search, "r_format": r_format, "r_answer": r_answer, "n_search": n}
    if reasoning_judge is not None and reasoning_weight > 0:
        out["r_reasoning"] = reasoning_judge.score_reasoning(question, response)
        out["reward"] += reasoning_weight * out["r_reasoning"]
    return out
