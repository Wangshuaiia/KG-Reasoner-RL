"""Build the SFT warm-up set (Sec. 4.1 / 4.2.3 / Appendix A): ~500 cold-start trajectories
in the <think>/<search>/<triples>/<answer> format, plus error-correction pairs.

Teachers:
  --teacher path      (default) search the KG with the real retrieval tool for a path from the
                      topic entity to a gold answer, then render it as a reasoning trajectory.
  --teacher hf:PATH   run an open-weight model (transformers) with the paper's warm-up prompt
  --teacher vllm:PATH (Appendix C) inside the tool loop; keep well-formed, correct trajectories.

Error correction (Sec. 4.2.3): for a fraction of correct trajectories, an error is injected at a
random step (searching a plausible but wrong entity), followed by its detection, a
"<search> Backtrack=True </search>" call and the revised (original) continuation. The tool is
re-run for every step, so all <triples> blocks are real tool outputs.

    python scripts/build_warmup_data.py --train_file data/rl/train.jsonl --n 500 --out data/sft/warmup.jsonl \
        --kg_backend remote --kg_server_url http://127.0.0.1:5003
"""
import argparse
import logging
import os
import random
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from kg_reasoner.data import load_records, write_jsonl  # noqa: E402
from kg_reasoner.kg import add_kg_args, tool_factory_from_args  # noqa: E402
from kg_reasoner.kg.base import norm_entity  # noqa: E402
from kg_reasoner.prompts import (ANSWER_CLOSE, ANSWER_OPEN, BACKTRACK_CALL, SEARCH_CLOSE, SEARCH_OPEN,  # noqa: E402
                                 THINK_CLOSE, build_prompt, build_warmup_prompt, format_observation)
from kg_reasoner.rewards import answer_match, check_format, extract_answer  # noqa: E402

logger = logging.getLogger("warmup")

OPEN = [
    "I need to find information about {e} to answer this question. I will search the knowledge graph for {e}.",
    "I am not sure about the facts needed here, so I will start from the topic entity {e}.",
    "To answer the question, I first need to know more about {e}. Let me search for it.",
]
HOP = [
    "The triple {t} shows that {a} is connected to {b}. Next, I need information about {b}.",
    "From the retrieved triples, {t} links {a} to {b}, which is relevant to the question. I will search {b}.",
    "Based on {t}, the next entity on the reasoning path is {b}. Let me search {b}.",
]
FINAL = [
    "The triple {t} answers the question: the answer is {ans}.",
    "Based on the search results, {t} shows that the answer is {ans}.",
]
FACT = "Based on the retrieved triples about {e} and my knowledge, the claim is {ans}."
WRONG = [
    "The triple {t} suggests that {d} might be relevant. I will search {d}.",
    "{d} appears in {t}; it may lead to the answer. Let me search {d}.",
]
DETECT = [
    "The triples about {d} are not related to what the question asks, so {d} is a dead end. I need to backtrack.",
    "None of these triples help to answer the question; searching {d} was a mistake. I will backtrack.",
]
RECOVER = "After backtracking to {a}, the triple {t} is the relevant one."


def find_path(tool, rec, max_hops):
    """BFS over the triples the tool actually shows. Returns [(entity, link_triple), ...] where the
    last link triple contains the answer."""
    name_ids = getattr(tool.kg, "name_ids", False)
    queue = deque((te["name"], te["name"] if name_ids else te["id"], []) for te in rec["topic_entities"])
    seen = set()
    while queue:
        name, eid, path = queue.popleft()
        key = norm_entity(name)
        if key in seen or len(path) >= max_hops or not eid:
            continue
        seen.add(key)
        ancestors = {norm_entity(e) for e, _ in path} | {key}
        shown = []
        for tr, _ in tool.ranked(name, eid):
            if norm_entity(tr.other_end(key)[0]) not in ancestors:
                shown.append(tr)
            if len(shown) >= tool.top_k:
                break
        for tr in shown:
            other, oid = tr.other_end(key)
            if answer_match(other, rec["answers"], strict=True):
                return path + [(name, tr)], other
        for tr in shown:
            other, oid = tr.other_end(key)
            queue.append((other, oid, path + [(name, tr)]))
    return None, None


def render(tool_factory, rec, path, answer, inject_at=None, rng=random):
    """Replay `path` through a fresh tool and write the trajectory text (after "<think>")."""
    tool = tool_factory(rec)
    text = " " + rng.choice(OPEN).format(e=path[0][0])
    n_search = 0
    for i, (entity, link) in enumerate(path):
        text += f"\n{SEARCH_OPEN}{entity}{SEARCH_CLOSE}" + format_observation(tool.search(entity))
        n_search += 1
        if inject_at == i:
            key = norm_entity(entity)
            wrong = [t for t in tool.tracker.candidates(tool.tracker.nodes[key], tool.top_k)
                     if t != link and not answer_match(t.other_end(key)[0], rec["answers"], strict=True)]
            for t in wrong:
                d = t.other_end(key)[0]
                obs = tool.search(d)
                if answer_match(obs, rec["answers"]):  # the "wrong" branch would actually reach the answer
                    continue
                text += rng.choice(WRONG).format(t=t.render(), d=d)
                text += f"\n{SEARCH_OPEN}{d}{SEARCH_CLOSE}" + format_observation(obs)
                text += rng.choice(DETECT).format(d=d)
                text += f"\n{SEARCH_OPEN}{BACKTRACK_CALL}{SEARCH_CLOSE}" + format_observation(tool.search("", True))
                text += RECOVER.format(a=entity, t=link.render()) + " "
                n_search += 2
                break
            else:
                return None
        if i + 1 < len(path):
            text += rng.choice(HOP).format(t=link.render(), a=entity, b=path[i + 1][0])
        else:
            text += rng.choice(FINAL).format(t=link.render(), ans=answer)
    text += f"\n{THINK_CLOSE}\n{ANSWER_OPEN}{answer}{ANSWER_CLOSE}"
    return text, n_search


def render_fact_check(tool_factory, rec, rng=random):
    te = rec["topic_entities"][0]["name"]
    tool = tool_factory(rec)
    text = " " + rng.choice(OPEN).format(e=te)
    text += f"\n{SEARCH_OPEN}{te}{SEARCH_CLOSE}" + format_observation(tool.search(te))
    ans = rec["answers"][0]
    text += FACT.format(e=te, ans=ans) + f"\n{THINK_CLOSE}\n{ANSWER_OPEN}{ans}{ANSWER_CLOSE}"
    return text, 1


def path_teacher(rec, tool_factory, args, seed):
    rng = random.Random(seed)
    if not rec.get("topic_entities"):
        return []
    try:
        if rec.get("task") == "fact_checking":
            out = render_fact_check(tool_factory, rec, rng)
            return [_row(rec, out, "path")] if out else []
        path, answer = find_path(tool_factory(rec), rec, args.max_hops)
        if not path:
            return []
        rows = [_row(rec, render(tool_factory, rec, path, answer, rng=rng), "path")]
        if rng.random() < args.error_ratio:
            out = render(tool_factory, rec, path, answer, inject_at=rng.randrange(len(path)), rng=rng)
            if out:
                rows.append(_row(rec, out, "error_correction"))
        return rows
    except Exception:
        logger.exception("failed on %s", rec["id"])
        return []


def _row(rec, out, kind):
    response, n_search = out
    return {"id": rec["id"], "dataset": rec["dataset"], "question": rec["question"], "answers": rec["answers"],
            "prompt": build_prompt(rec), "response": response, "type": kind, "n_search": n_search}


def model_teacher(records, tool_factory, args):
    """Appendix C warm-up prompt with an open-weight teacher model in the tool loop."""
    from transformers import AutoTokenizer

    from kg_reasoner.rollout import HFGenerator, RolloutConfig, VLLMGenerator, make_episode, run_episodes

    kind, path = args.teacher.split(":", 1)
    tok = AutoTokenizer.from_pretrained(path)
    if kind == "vllm":
        from vllm import LLM

        gen = VLLMGenerator(LLM(model=path, trust_remote_code=True), temperature=args.temperature)
    else:
        from transformers import AutoModelForCausalLM

        from kg_reasoner.training import pick_device

        model = AutoModelForCausalLM.from_pretrained(path).to(pick_device(args.device)).eval()
        gen = HFGenerator(model, tok, temperature=args.temperature)
    eps = [make_episode(r, tok, tool_factory, prompt=build_warmup_prompt(r)) for r in records]
    run_episodes(eps, gen, tok, RolloutConfig(max_turns=args.max_turns))
    rows = []
    for ep in eps:
        resp = ep.response_text
        if check_format(resp) and answer_match(extract_answer(resp), ep.record["answers"]):
            rows.append(_row(ep.record, (resp, ep.n_search), "teacher"))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_file", required=True)
    p.add_argument("--out", default="data/sft/warmup.jsonl")
    p.add_argument("--n", type=int, default=500, help="number of correct trajectories to keep")
    p.add_argument("--teacher", default="path", help="path | hf:MODEL | vllm:MODEL")
    p.add_argument("--error_ratio", type=float, default=0.3, help="share of trajectories that get an error-correction pair")
    p.add_argument("--max_hops", type=int, default=3)
    p.add_argument("--max_turns", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--device", default="auto")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max_attempts", type=int, default=None, help="questions to try (default 3*n)")
    p.add_argument("--seed", type=int, default=0)
    add_kg_args(p)
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)

    records = load_records(args.train_file)
    random.Random(args.seed).shuffle(records)
    records = records[: args.max_attempts or 3 * args.n]
    tool_factory = tool_factory_from_args(args)

    if args.teacher == "path":
        with ThreadPoolExecutor(args.workers) as pool:
            results = list(pool.map(lambda x: path_teacher(x[1], tool_factory, args, args.seed + x[0]), enumerate(records)))
        rows, kept = [], 0
        for rs in results:
            if kept >= args.n:
                break
            if rs:
                rows += rs
                kept += 1
    else:
        rows = model_teacher(records, tool_factory, args)
        rows = rows[: args.n]

    bad = [r for r in rows if not check_format(r["response"])]
    assert not bad, f"{len(bad)} malformed trajectories, e.g. {bad[0]['response'][:500]}"
    write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} trajectories to {args.out}:",
          {k: sum(r["type"] == k for r in rows) for k in ("path", "error_correction", "teacher")})


if __name__ == "__main__":
    main()
