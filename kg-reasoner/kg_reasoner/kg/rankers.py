"""Entity rankers used by the retrieval tool to keep the top-k triples (Sec. 4.2.2).

- GNNRanker:        the paper's GNN-based entity ranking (default).
- SimilarityRanker: untrained encoder similarity (baseline / no checkpoint available).
"""
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, List, Tuple

from .base import KnowledgeGraph, Triple, norm_entity
from .gnn import GNNEntityScorer, TextEncoder, build_batch, relation_text

Ranked = List[Tuple[Triple, float]]


class ContextProvider:
    """Neighbors of candidate entities, used as GNN context.
    mode: "none" | "cached" (only neighborhoods already fetched, no new lookups) |
          "fetch" (look up at most `max_fetch` candidates per ranking call)."""

    def __init__(self, kg: KnowledgeGraph, mode: str = "cached", max_fetch: int = 5):
        self.kg, self.mode, self.max_fetch = kg, mode, max_fetch

    def __call__(self, cands: List[Tuple[str, str]], kg_style: str) -> List[List[Triple]]:
        out, fetched = [], 0
        for name, eid in cands:
            triples = None
            if self.mode in ("cached", "fetch"):
                triples = self.kg.cached_neighbors(eid or name, kg_style)
            if triples is None and self.mode == "fetch" and fetched < self.max_fetch and eid:
                triples, fetched = self.kg.neighbors(eid, name, kg_style), fetched + 1
            out.append(triples or [])
        return out


class Ranker(ABC):
    @abstractmethod
    def rank(self, question: str, entity_name: str, entity_id: str, kg_style: str,
             triples: List[Triple]) -> Ranked:
        """Return triples sorted by relevance (best first) with scores."""


def _group_by_candidate(entity_name: str, triples: List[Triple]):
    key = norm_entity(entity_name)
    groups: "OrderedDict[str, Dict]" = OrderedDict()
    for tr in triples:
        name, eid = tr.other_end(key)
        g = groups.setdefault(norm_entity(name), {"name": name, "id": eid, "triples": []})
        g["triples"].append(tr)
    return groups


class SimilarityRanker(Ranker):
    def __init__(self, encoder: TextEncoder):
        self.encoder = encoder

    def rank(self, question, entity_name, entity_id, kg_style, triples) -> Ranked:
        if not triples:
            return []
        q = self.encoder.encode([question])
        texts = [f"{tr.head} {relation_text(tr.relation)} {tr.tail}" for tr in triples]
        scores = (self.encoder.encode(texts) @ q.T).squeeze(-1).tolist()
        return sorted(zip(triples, scores), key=lambda x: -x[1])


def candidate_inputs(entity_name: str, kg_style: str, triples: List[Triple], context: ContextProvider):
    """Group the searched entity's triples by candidate (other end) and attach each candidate's
    one-hop neighbors (relation, neighbor) as GNN context. Returns (groups, candidates, contexts)."""
    groups = _group_by_candidate(entity_name, triples)
    contexts = context([(g["name"], g["id"]) for g in groups.values()], kg_style)
    cands = []
    for g, ctx in zip(groups.values(), contexts):
        ckey = norm_entity(g["name"])
        neigh = [(tr.relation, entity_name) for tr in g["triples"]]  # linking edge(s) to the searched entity
        neigh += [(tr.relation, tr.other_end(ckey)[0]) for tr in ctx]
        cands.append((g["name"], neigh))
    return groups, cands, contexts


class GNNRanker(Ranker):
    def __init__(self, model: GNNEntityScorer, encoder: TextEncoder, context: ContextProvider,
                 max_neighbors: int = 32):
        self.model, self.encoder, self.context, self.max_neighbors = model, encoder, context, max_neighbors

    def rank(self, question, entity_name, entity_id, kg_style, triples) -> Ranked:
        if not triples:
            return []
        groups, cands, _ = candidate_inputs(entity_name, kg_style, triples, self.context)
        device = next(self.model.parameters()).device
        batch = [x.to(device) for x in build_batch(self.encoder, question, cands, self.max_neighbors)]
        scores = self.model.score(*batch).tolist()
        ranked = [(tr, s) for g, s in zip(groups.values(), scores) for tr in g["triples"]]
        return sorted(ranked, key=lambda x: -x[1])
