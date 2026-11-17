"""Offline tests (no API calls, no GPU):  python -m pytest -q tests"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kg_reasoner.data import load_records
from kg_reasoner.kg.base import KnowledgeGraph, Triple
from kg_reasoner.kg.rankers import Ranker
from kg_reasoner.kg.tool import KGRetrievalTool, parse_search_query
from kg_reasoner.prompts import build_prompt, format_observation, split_query, split_response_segments
from kg_reasoner.rewards import answer_match, check_format, compute_reward, count_searches

ROOT = os.path.join(os.path.dirname(__file__), "..")

# 1 -> 2 -> 4 -> 5 (Fig. 3); 2 also links to 3, 4 also links to 6
GRAPH = {
    "E1": [("E1", "r12", "E2")],
    "E2": [("E1", "r12", "E2"), ("E2", "r24", "E4"), ("E2", "r23", "E3")],
    "E3": [("E2", "r23", "E3"), ("E3", "r3a", "Answer")],
    "E4": [("E2", "r24", "E4"), ("E4", "r45", "E5"), ("E4", "r46", "E6")],
    "E5": [("E4", "r45", "E5")],
    "E6": [("E4", "r46", "E6")],
}


class FakeKG(KnowledgeGraph):
    name_ids = True

    def neighbors(self, entity_id, entity_name=None, kg_style="freebase"):
        return [Triple(h, r, t, h, t) for h, r, t in GRAPH.get(entity_id, [])]


class OrderRanker(Ranker):
    """Prefers r45 over r46 and r24 over r23, so the first path is the wrong one."""

    PRIORITY = {"r24": 0, "r23": 1, "r45": 0, "r46": 1}

    def rank(self, question, entity_name, entity_id, kg_style, triples):
        return sorted([(t, -self.PRIORITY.get(t.relation, 5)) for t in triples], key=lambda x: -x[1])


def make_tool(k=1):
    return KGRetrievalTool(FakeKG(), OrderRanker(), "q?", [{"id": "E1", "name": "E1"}], top_k=k)


def test_parse_search_query():
    assert parse_search_query(" Utah ") == ("Utah", False)
    assert parse_search_query("Backtrack=True") == ("", True)
    assert parse_search_query("Lou Seal, Backtrack=True") == ("Lou Seal", True)
    assert parse_search_query('"Lou Seal"') == ("Lou Seal", False)


def test_backtracking_follows_fig3():
    tool = make_tool(k=1)
    assert "E2" in tool("E1")
    assert "E4" in tool("E2")                      # r24 preferred over r23
    assert "E5" in tool("E4")                      # r45 preferred over r46
    assert "No triples" in tool("E5")              # 5 cannot support further reasoning
    out = tool("Backtrack=True")                   # back to parent 4 -> alternative 6
    assert out.startswith("Backtracked from E5 to E4") and "E6" in out
    tool("E6")
    out = tool("Backtrack=True")                   # 4 has nothing left -> climb to 2 -> alternative 3
    assert out.startswith("Backtracked from E6 to E2") and "E3" in out
    assert "Answer" in tool("E3")
    assert tool.tracker.path_names() == ["E1", "E2", "E3"]


def test_backtrack_reaches_topic_entity():
    tool = make_tool(k=5)
    tool("E1")
    tool("E2")
    tool.tracker.nodes["e2"].dead = True
    out = tool("E2, Backtrack=True")
    assert "returned to the topic entity" in out


def make_response(n_search=1, answer="Answer"):
    resp = ""
    for i in range(n_search):
        resp += f" I will search E{i}. <search>E{i}</search>" + format_observation(f"(E{i}, r, E{i + 1})")
    return resp + f"So the answer is {answer}.\n</think>\n<answer>{answer}</answer>"


def test_format_and_rewards():
    good = make_response(2)
    assert check_format(good) and count_searches(good) == 2
    r = compute_reward(good, ["Answer"])
    assert r == {"reward": 0.8 + 0.5 + 1.0, "r_search": 0.8, "r_format": 0.5, "r_answer": 1.0, "n_search": 2}
    r1 = compute_reward(make_response(1, "wrong"), ["Answer"])
    assert (r1["r_search"], r1["r_format"], r1["r_answer"]) == (0.5, 0.5, 0.0)
    assert not check_format("no tags at all")
    assert not check_format(good.replace("</triples>", "", 1))                    # broken triples block
    assert not check_format(good + " trailing text")
    assert not check_format("<search>x</search> </think><answer>a</answer>")       # search without tool output
    assert compute_reward("</think><answer>Answer</answer>", ["Answer"])["reward"] == 1.5


def test_answer_match():
    assert answer_match("Mountain Time Zone", ["Mountain Time Zone"])
    assert answer_match("It is the 2014 World Series", ["2014 World Series"])
    assert answer_match("True", ["true"]) and not answer_match("false", ["true"])
    assert not answer_match("", ["x"])
    assert answer_match("United States-Mexico-Canada Agreement", ["Canada"])
    assert not answer_match("United States-Mexico-Canada Agreement", ["Canada"], strict=True)


def test_prompt_roundtrip_and_segments():
    rec = {"question": "What timezone is Utah in?", "topic_entities": [{"id": "Q829", "name": "Utah"}], "task": "qa"}
    prompt = build_prompt(rec)
    assert prompt.endswith("Assistant: <think>")
    resp = make_response(1)
    q, te, r = split_query(prompt + resp + "<|endoftext|>")
    assert q == rec["question"] and te == "Utah" and r.startswith(" I will search")
    segs = split_response_segments(resp)
    assert [g for _, g in segs] == [True, False, True] and "".join(t for t, _ in segs) == resp


def test_all_eight_datasets_load():
    names = ["cwq", "WebQSP", "WebQuestions", "graliqa", "qald_10-en", "T-REX", "Zero_Shot_RE", "creak"]
    for n in names:
        recs = load_records(os.path.join(ROOT, "data", "test_set", f"{n}.json"), limit=5)
        assert recs, n
        for r in recs:
            assert r["question"] and r["answers"] and r["kg"] in ("freebase", "wikidata"), (n, r)


class CharTokenizer:
    """Minimal tokenizer for tests: one id per character, eos = 0."""

    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids if i)


def test_rollout_masks_and_packing():
    from kg_reasoner.rollout import GenOutput, RolloutConfig, make_episode, pack_episodes, run_episodes

    tok = CharTokenizer()
    script = [" Search E1. <search>E1</search>", " Found it.\n</think>\n<answer>Answer</answer>"]

    def generate(batch_ids, budgets):
        outs = []
        for ids in batch_ids:
            text = script[0] if "(E1, r12, E2)" not in tok.decode(ids) else script[1]
            stop = "</search>" if text.endswith("</search>") else "</answer>"
            outs.append(GenOutput(tok(text)["input_ids"], text, stop))
        return outs

    rec = {"question": "q?", "answers": ["Answer"], "topic_entities": [{"id": "E1", "name": "E1"}], "kg": "freebase"}
    tool_factory = lambda r: KGRetrievalTool(FakeKG(), OrderRanker(), r["question"], r["topic_entities"], top_k=3)
    eps = [make_episode(rec, tok, tool_factory) for _ in range(2)]
    run_episodes(eps, generate, tok, RolloutConfig(max_turns=4))
    ep = eps[0]
    assert ep.finish_reason == "answer" and ep.n_search == 1
    assert check_format(ep.response_text)
    assert compute_reward(ep.response_text, rec["answers"])["reward"] == 0.5 + 0.5 + 1.0
    # tool output (and only tool output) is masked
    masked = "".join(chr(t) for t, m in zip(ep.response_ids, ep.response_mask) if m == 0)
    assert masked.startswith("\n<triples>\n(E1, r12, E2)") and masked.endswith("</triples>\n")
    assert ep.response_ids[-1] == tok.eos_token_id and ep.response_mask[-1] == 1
    packed = pack_episodes(eps)
    assert len(packed["sequences"]) == sum(packed["packed_seq_lens"]) == len(packed["attention_mask"])
    assert len(packed["retrieve_mask"]) == sum(packed["num_actions"])
    assert packed["retrieve_num"] == [1, 1]


def test_unified_jsonl_roundtrip(tmp_path):
    from kg_reasoner.data import write_jsonl

    recs = []
    for n in ["cwq", "creak", "T-REX", "qald_10-en"]:
        recs += load_records(os.path.join(ROOT, "data", "test_set", f"{n}.json"), limit=2)
    path = str(tmp_path / "rl.jsonl")
    write_jsonl(path, recs)
    assert load_records(path) == recs


def test_split_query_ignores_imitated_prompt_in_response():
    rec = {"question": "Real question?", "topic_entities": [{"id": "x", "name": "X"}], "task": "qa"}
    fake = "\nUser: Fake?\n  Topic Entity: Y\nAssistant: <think> ..."
    q, te, resp = split_query(build_prompt(rec) + " hmm" + fake)
    assert q == "Real question?" and te == "X" and resp.startswith(" hmm")
