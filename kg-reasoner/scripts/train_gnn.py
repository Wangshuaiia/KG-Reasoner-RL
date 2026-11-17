"""Train the GNN entity ranker (paper Sec. 4.2.2, cross-entropy of Eq. 4).

For each training question the topic entity's one-hop candidates are labelled relevant when the
candidate is a gold answer, or when a gold answer is among the candidate's own neighbors
(i.e. it lies on a 2-hop path to the answer). Text is embedded by a frozen sentence encoder.

    python scripts/train_gnn.py --train_file data/rl/train.jsonl --out ckpt/gnn.pt \
        --sparql_endpoint http://localhost:8890/sparql
"""
import argparse
import logging
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from kg_reasoner.data import load_records  # noqa: E402
from kg_reasoner.kg import add_kg_args, build_kg  # noqa: E402
from kg_reasoner.kg.base import norm_entity  # noqa: E402
from kg_reasoner.kg.gnn import DEFAULT_ENCODER, GNNEntityScorer, TextEncoder, build_batch, save_gnn  # noqa: E402
from kg_reasoner.kg.rankers import ContextProvider, candidate_inputs  # noqa: E402
from kg_reasoner.rewards import answer_match  # noqa: E402


def collect(rec, kg, context, max_candidates):
    if not rec.get("topic_entities") or rec.get("task") == "fact_checking":
        return None
    te = rec["topic_entities"][0]
    eid = te["name"] if kg.name_ids else te["id"]
    triples = kg.neighbors(eid, te["name"], rec["kg"])
    if not triples:
        return None
    groups, cands, contexts = candidate_inputs(te["name"], rec["kg"], triples, context)
    labels = []
    for g, ctx in zip(groups.values(), contexts):
        ckey = norm_entity(g["name"])
        two_hop = any(answer_match(t.other_end(ckey)[0], rec["answers"], strict=True) for t in ctx)
        labels.append(int(answer_match(g["name"], rec["answers"], strict=True) or two_hop))
    keep = list(range(len(cands)))[:max_candidates]
    return rec["question"], [cands[i] for i in keep], [labels[i] for i in keep]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_file", required=True)
    p.add_argument("--out", default="ckpt/gnn.pt")
    p.add_argument("--max_questions", type=int, default=500)
    p.add_argument("--max_candidates", type=int, default=40)
    p.add_argument("--fetch_context", type=int, default=5, help="candidate neighborhoods looked up per question")
    p.add_argument("--neg_ratio", type=int, default=3, help="negatives kept per positive")
    p.add_argument("--encoder", default=DEFAULT_ENCODER)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    add_kg_args(p)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    recs = load_records(args.train_file)[: args.max_questions]
    kg = build_kg(args.kg_backend, sparql_endpoint=args.sparql_endpoint, server_url=args.kg_server_url)
    context = ContextProvider(kg, mode="fetch", max_fetch=args.fetch_context)
    with ThreadPoolExecutor(args.workers) as pool:
        data = [d for d in pool.map(lambda r: collect(r, kg, context, args.max_candidates), recs) if d]

    encoder = TextEncoder(args.encoder)
    samples = []
    for question, cands, labels in data:
        pos = [i for i, y in enumerate(labels) if y]
        neg = [i for i, y in enumerate(labels) if not y]
        neg = random.sample(neg, min(len(neg), max(args.neg_ratio * len(pos), args.neg_ratio)))
        idx = pos + neg
        if idx:
            samples.append((question, [cands[i] for i in idx], [labels[i] for i in idx]))
    n_pos = sum(sum(s[2]) for s in samples)
    n_all = sum(len(s[2]) for s in samples)
    print(f"{len(samples)} questions, {n_all} candidates, {n_pos} relevant")
    if not samples:
        sys.exit("No training candidates were collected (check the KG backend).")
    if n_pos == 0:
        print("[warning] no relevant candidates found; the ranker will not learn anything useful")

    model = GNNEntityScorer(encoder.dim, hidden=args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    batches = [(build_batch(encoder, q, c), torch.tensor(y)) for q, c, y in samples]
    for epoch in range(args.epochs):
        random.shuffle(batches)
        total, correct, loss_sum = 0, 0, 0.0
        for inputs, y in batches:
            logits = model(*inputs)
            loss = F.cross_entropy(logits, y)  # Eq. 4
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(-1) == y).sum().item()
            total += len(y)
        print(f"epoch {epoch + 1}: loss {loss_sum / total:.4f}  acc {correct / total:.3f}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_gnn(args.out, model.eval(), args.encoder)
    print("saved", args.out)


if __name__ == "__main__":
    main()
