"""The custom KG retrieval tool triggered by </search> (paper Sec. 4.1-4.2).

search(entity, Backtrack=False):
    one-hop neighbor retrieval (4.2.1) -> entity ranking (4.2.2) -> top-k triples.
search(entity, Backtrack=True):
    backtracking along the stored retrieval path (4.2.3).
One tool instance is created per reasoning episode (it owns the path state).
"""
import re
from typing import Dict, List, Optional, Tuple

from .backtrack import PathTracker
from .base import KnowledgeGraph, Triple, norm_entity
from .rankers import Ranked, Ranker

_BACKTRACK_RE = re.compile(r"[\s,;|]*\bbacktrack\s*[=:]\s*(true|false|yes|no|1|0)\b[\s,;|]*", re.I)


def parse_search_query(raw: str) -> Tuple[str, bool]:
    """'Utah' -> ('Utah', False); 'Backtrack=True' -> ('', True); 'Lou Seal, Backtrack=True' -> ('Lou Seal', True)."""
    text, backtrack = raw.strip(), False
    m = _BACKTRACK_RE.search(text)
    if m:
        backtrack = m.group(1).lower() in ("true", "yes", "1")
        text = _BACKTRACK_RE.sub(" ", text)
    elif text.lower().strip("[]() ") == "backtrack":
        backtrack, text = True, ""
    text = re.sub(r"^\s*(entity|query|search)\s*:\s*", "", text, flags=re.I)
    text = " ".join(text.strip().strip("\"'`[]()").split())
    return text, backtrack


def render(triples: List[Triple]) -> str:
    return "\n".join(t.render() for t in triples)


class KGRetrievalTool:
    def __init__(self, kg: KnowledgeGraph, ranker: Ranker, question: str, topic_entities: List[Dict],
                 kg_style: str = "freebase", top_k: int = 5, max_triples: int = 300,
                 rank_cache: Optional[Dict] = None):
        self.kg, self.ranker, self.question = kg, ranker, question
        self.kg_style, self.top_k, self.max_triples = kg_style, top_k, max_triples
        name_ids = getattr(kg, "name_ids", False)
        topics = [(te["name"], te["name"] if name_ids else te["id"]) for te in topic_entities if te.get("name")]
        self.tracker = PathTracker(topics)
        self.rank_cache = {} if rank_cache is None else rank_cache
        self.calls: List[Dict] = []

    def __call__(self, raw_query: str) -> str:
        entity, backtrack = parse_search_query(raw_query)
        return self.search(entity, backtrack=backtrack)

    def ranked(self, name: str, eid: str) -> Ranked:
        key = (self.question, self.kg_style, norm_entity(name), eid)
        if key not in self.rank_cache:
            triples = self.kg.neighbors(eid, name, self.kg_style)[: self.max_triples]
            self.rank_cache[key] = self.ranker.rank(self.question, name, eid, self.kg_style, triples)
        return self.rank_cache[key]

    def search(self, entity: str, backtrack: bool = False) -> str:
        if backtrack:
            out, node_name = self._backtrack(entity)
        else:
            out, node_name = self._search(entity)
        self.calls.append({"entity": entity, "backtrack": backtrack, "node": node_name, "output": out})
        return out

    def _search(self, entity: str) -> Tuple[str, Optional[str]]:
        if not entity:
            return "Invalid search: the query must contain exactly one entity.", None
        hit = self.tracker.resolve(entity)
        if hit is not None:
            name, eid, parent = hit
        else:  # not a topic entity / not in retrieved triples: still try the KG
            name, eid, parent = entity, self.kg.resolve(entity, self.kg_style), self.tracker.current
        if not eid:
            return f"No entity named '{entity}' was found in the knowledge graph.", None
        node = self.tracker.visit(name, eid, parent, self.ranked(name, eid))
        shown = self.tracker.candidates(node, self.top_k)
        self.tracker.register_shown(node.key, shown)
        return (render(shown) if shown else f"No triples found for {name}."), name

    def _backtrack(self, entity: str) -> Tuple[str, Optional[str]]:
        dead_key = self.tracker.key_of(entity) if entity else None
        node, alts, dead_name = self.tracker.backtrack(dead_key, self.top_k)
        if node is not None:
            self.tracker.register_shown(node.key, alts)
            return f"Backtracked from {dead_name} to {node.name}.\n{render(alts)}", node.name
        if dead_name is None:
            return "Nothing to backtrack: no entity has been searched yet.", None
        roots = self.tracker.unexplored_roots()
        if roots:
            return (f"No alternative candidates remain on the current path. "
                    f"Unexplored topic entities: {', '.join(n for n, _ in roots)}."), None
        return "No alternative candidates remain; the search has returned to the topic entity.", None
