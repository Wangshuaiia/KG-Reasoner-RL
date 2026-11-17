"""Qwen-Agent + Qwen2.5-7B-Instruct (4-bit, MLX) with a real-KG integration interface.

The agent is Qwen-Agent's `Assistant` with a single tool, `kg_search` (kg_tool.py), which returns
one-hop triples through an external KnowledgeGraph implementation (kg_interface.py).
No KG implementation or data is bundled with this harness. The model calls the tool with Qwen2.5's
native <tool_call> format (parsed by mlx_lm.server into OpenAI `tool_calls`), follows the entities it
finds hop by hop, and ends with "Final answer: ...".
Scoring is Hits@1 with KG-Reasoner's answer matching, on the same questions as
../qwen2.5-7b-4bit/kbqa_no_kg.py (same seed), so the two runs can be compared question by question.

    # terminal 1: serve the model through an OpenAI-compatible API
    ../.venv/bin/python -m mlx_lm server --model mlx-community/Qwen2.5-7B-Instruct-4bit --port 8080
    # terminal 2
    ../.venv/bin/python run_agent_kbqa.py --kg_backend my_kg:RealKG --datasets cwq WebQSP --n 20
"""
import argparse
import json
import os
import random
import re
import sys
import time

os.environ.setdefault("QWEN_AGENT_MAX_LLM_CALL_PER_RUN", "8")  # read by qwen_agent.settings at import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "kg-reasoner"))
sys.path.insert(0, REPO)
from kg_reasoner.data import load_records  # noqa: E402
from kg_reasoner.rewards import answer_match  # noqa: E402

from kg_interface import load_knowledge_graph  # noqa: E402

MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
NO_KG_RESULTS = os.path.join(HERE, "..", "qwen2.5-7b-4bit", "results", "kbqa_no_kg.jsonl")

SYSTEM = """You are a question-answering agent. You can look up facts in a knowledge graph with the kg_search tool.

How to work:
1. Start from a topic entity of the question and look it up with kg_search.
2. Many questions need several hops. Pick the entity in the result that leads toward the answer and look it up next.
3. If a lookup returns nothing useful, try another entity or ask for a different relation; do not repeat the same call.
4. Base the answer on the retrieved facts. Use your own knowledge only when the knowledge graph has nothing relevant.
5. When you have the answer, reply briefly. If several entities qualify, choose the single one you are most confident about.
The last line of your reply must be:
Final answer: <one answer entity>"""

FINAL_RE = re.compile(r"final answer\s*[:：]\s*(.+)", re.I)


def first_entity(pred: str) -> str:
    """Hits@1 scores one answer: if the model still lists several, keep the first."""
    return re.split(r"\s*(?:;|, | / |\|)\s*", pred or "", maxsplit=1)[0].strip()


def extract_final(text: str):
    found = FINAL_RE.findall(text or "")
    if found:
        ans = found[-1]
    else:
        lines = [ln for ln in (text or "").strip().splitlines() if ln.strip()]
        ans = lines[-1] if lines else ""
    return ans.strip().strip("*").strip().rstrip("."), bool(found)


def load_no_kg(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return {(r["dataset"], str(r["id"])): r for r in rows}


def run_one(rec, kg, llm_cfg, top_k):
    from qwen_agent.agents import Assistant
    from kg_tool import KGSearch

    tool = KGSearch(kg, rec["question"], top_k=top_k)
    bot = Assistant(llm=llm_cfg, function_list=[tool], system_message=SYSTEM)
    names = ", ".join(e["name"] for e in rec["topic_entities"] if e.get("name"))
    user = f"Question: {rec['question']}\nTopic entities: {names or 'unknown'}"
    t0, error = time.time(), ""
    try:
        messages = bot.run_nonstream([{"role": "user", "content": user}])
    except Exception as ex:  # keep going; the failure is recorded
        messages, error = [], f"{type(ex).__name__}: {ex}"
    final_text = ""
    if messages and messages[-1].get("role") == "assistant" and not messages[-1].get("function_call"):
        final_text = messages[-1].get("content") or ""
        if isinstance(final_text, list):
            final_text = "".join(c.get("text", "") for c in final_text)
    pred, formatted = extract_final(final_text)
    retrieved = {x for step in tool.trace for (h, _, t) in step["triples"] for x in (h, t)}
    return {
        "pred": pred,
        "hit": answer_match(first_entity(pred), rec["answers"]),
        "hit_any_listed": answer_match(pred, rec["answers"]),
        "final_answer_line": formatted,
        "ran_out_of_calls": bool(messages) and not final_text,
        "error": error,
        "n_search": len(tool.trace),
        "answer_entity_retrieved": any(answer_match(x, rec["answers"], strict=True) for x in retrieved),
        "searches": [{"entity": s["entity"], "relation": s["relation"], "resolved": s["resolved"],
                      "n_triples": len(s["triples"])} for s in tool.trace],
        "sec": round(time.time() - t0, 1),
        "trajectory": [{k: m.get(k) for k in ("role", "content", "function_call", "name") if m.get(k)}
                       for m in messages],
    }


def pct(xs):
    return round(100 * sum(xs) / max(1, len(xs)), 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["cwq", "WebQSP"], help="file names in kg-reasoner/data/test_set")
    p.add_argument("--n", type=int, default=20, help="questions sampled per dataset")
    p.add_argument("--seed", type=int, default=0, help="same seed as kbqa_no_kg.py gives the same questions")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--server", default="http://127.0.0.1:8080/v1")
    p.add_argument("--max_tokens", type=int, default=768, help="generation budget per LLM call")
    p.add_argument("--top_k", type=int, default=10, help="triples returned per kg_search call")
    p.add_argument("--kg_backend", required=True, metavar="MODULE:CLASS",
                   help="external real-KG class implementing kg_interface.KnowledgeGraph")
    p.add_argument("--kg_config", help="JSON file containing KG backend constructor keyword arguments")
    p.add_argument("--no_kg_results", default=NO_KG_RESULTS,
                   help="per-question output of kbqa_no_kg.py with the same --n/--seed, for paired comparison")
    p.add_argument("--tag", default="qwen_agent_real_kg")
    p.add_argument("--out_dir", default=os.path.join(HERE, "results"))
    args = p.parse_args()
    if args.top_k < 1:
        p.error("--top_k must be positive")
    try:
        config = None
        if args.kg_config:
            with open(args.kg_config, encoding="utf-8") as f:
                config = json.load(f)
            if not isinstance(config, dict):
                raise ValueError("KG configuration must be a JSON object")
        kg = load_knowledge_graph(args.kg_backend, config)
    except (ImportError, AttributeError, TypeError, ValueError, OSError) as ex:
        p.error(f"Cannot initialize the real KG backend: {ex}")

    llm_cfg = {"model": args.model, "model_server": args.server, "api_key": "EMPTY",
               # use_raw_api: pass tools through the OpenAI `tools` field. mlx_lm.server renders them with
               # Qwen2.5's chat template and parses the model's <tool_call> output into `tool_calls`, so
               # Qwen-Agent's own text-level function-call prompting ("nous") must stay off here.
               "generate_cfg": {"use_raw_api": True, "temperature": 0.0, "seed": 0,
                                "max_tokens": args.max_tokens, "max_retries": 2}}
    no_kg = load_no_kg(args.no_kg_results)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.tag}.jsonl")
    t_start, summary = time.time(), {}
    with open(out_path, "w", encoding="utf-8") as fout:
        for name in args.datasets:
            records = load_records(os.path.join(REPO, "data", "test_set", f"{name}.json"))
            sample = random.Random(args.seed).sample(records, min(args.n, len(records)))
            rows = []
            for i, rec in enumerate(sample, 1):
                res = run_one(rec, kg, llm_cfg, args.top_k)
                base = no_kg.get((rec["dataset"], str(rec["id"])))
                row = {"dataset": rec["dataset"], "id": rec["id"], "hops": rec.get("hops"),
                       "question": rec["question"], "answers": rec["answers"][:10],
                       "no_kg_pred": base["pred"] if base else None,
                       "no_kg_hit": answer_match(first_entity(base["pred"]), rec["answers"]) if base else None,
                       **res}
                rows.append(row)
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                print(f"[{rec['dataset']} {i}/{len(sample)}] hit={int(row['hit'])} no_kg={row['no_kg_hit']} "
                      f"searches={row['n_search']} {row['sec']}s | pred={row['pred'][:60]!r} "
                      f"gold={row['answers'][:2]}{' | ' + row['error'] if row['error'] else ''}", flush=True)
            ds = sample[0]["dataset"] if sample else name
            paired = [r for r in rows if r["no_kg_hit"] is not None]
            summary[ds] = {
                "n": len(rows),
                "hits@1_agent_kg": pct([r["hit"] for r in rows]),
                "hit_any_listed_agent_kg": pct([r["hit_any_listed"] for r in rows]),
                "hits@1_no_kg_same_questions": pct([r["no_kg_hit"] for r in paired]) if paired else None,
                "fixed_by_kg": sum(1 for r in paired if r["hit"] and not r["no_kg_hit"]),
                "broken_by_kg": sum(1 for r in paired if r["no_kg_hit"] and not r["hit"]),
                "avg_searches": round(sum(r["n_search"] for r in rows) / max(1, len(rows)), 2),
                "pct_multi_search": pct([r["n_search"] >= 2 for r in rows]),
                "pct_answer_entity_retrieved": pct([r["answer_entity_retrieved"] for r in rows]),
                "pct_final_answer_line": pct([r["final_answer_line"] for r in rows]),
                "ran_out_of_calls": sum(r["ran_out_of_calls"] for r in rows),
                "errors": sum(bool(r["error"]) for r in rows),
                "sec_per_question": round(sum(r["sec"] for r in rows) / max(1, len(rows)), 1),
            }
            print(json.dumps({ds: summary[ds]}, indent=2), flush=True)

    meta = {"agent": "Qwen-Agent Assistant (native tool calls via mlx_lm.server)", "model": args.model,
            "kg_backend": args.kg_backend,
            "n_per_dataset": args.n, "seed": args.seed, "top_k": args.top_k,
            "max_llm_calls_per_question": int(os.environ["QWEN_AGENT_MAX_LLM_CALL_PER_RUN"]),
            "total_minutes": round((time.time() - t_start) / 60, 1), "results": summary}
    with open(os.path.join(args.out_dir, f"{args.tag}_summary.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
