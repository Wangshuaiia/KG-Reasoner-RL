"""HTTP clients for a shared KG server (kg_reasoner.kg.server), so that many rollout
processes share one KG connection, its neighbor cache and the GNN ranker."""
from typing import List, Optional

import requests

from .base import KnowledgeGraph, Triple
from .rankers import Ranked, Ranker


class _Client:
    def __init__(self, url: str, timeout: float = 300.0, retries: int = 3):
        self.url, self.timeout, self.retries = url.rstrip("/"), timeout, retries
        self.session = requests.Session()

    def post(self, path: str, payload: dict) -> dict:
        last = None
        for _ in range(self.retries):
            try:
                r = self.session.post(self.url + path, json=payload, timeout=self.timeout)
                r.raise_for_status()
                return r.json()
            except requests.RequestException as e:
                last = e
        raise RuntimeError(f"KG server {self.url}{path} failed: {last}")


class RemoteKG(KnowledgeGraph):
    def __init__(self, url: str):
        self.client = _Client(url)
        self.name_ids = bool(self.client.session.get(self.client.url + "/info", timeout=30).json()["name_ids"])

    def neighbors(self, entity_id, entity_name=None, kg_style="freebase") -> List[Triple]:
        out = self.client.post("/neighbors", {"entity_id": entity_id, "entity_name": entity_name, "kg_style": kg_style})
        return [Triple.from_dict(t) for t in out["triples"]]

    def cached_neighbors(self, entity_id, kg_style="freebase") -> Optional[List[Triple]]:
        out = self.client.post("/cached_neighbors", {"entity_id": entity_id, "kg_style": kg_style})
        return None if out["triples"] is None else [Triple.from_dict(t) for t in out["triples"]]

    def resolve(self, name, kg_style="freebase") -> Optional[str]:
        return self.client.post("/resolve", {"name": name, "kg_style": kg_style})["id"]


class RemoteRanker(Ranker):
    """Ranking runs on the server (which holds the GNN and the KG used for context)."""

    def __init__(self, url: str):
        self.client = _Client(url)

    def rank(self, question, entity_name, entity_id, kg_style, triples) -> Ranked:
        out = self.client.post("/rank", {"question": question, "entity_name": entity_name, "entity_id": entity_id,
                                         "kg_style": kg_style, "triples": [t.to_dict() for t in triples]})
        return [(Triple.from_dict(t), s) for t, s in zip(out["triples"], out["scores"])]
