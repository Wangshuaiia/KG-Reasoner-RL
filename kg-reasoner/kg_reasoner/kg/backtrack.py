"""Graph-based backtracking (paper Sec. 4.2.3, Fig. 3).

The tool keeps the traversed retrieval path. With a path 1 -> 2 -> 4 -> 5, a call with
Backtrack=True marks 5 as a dead end and returns alternative candidates from its parent 4
(ranked by relevance, excluding dead ends and ancestors). If 4 has no alternatives left, it
is marked as a dead end too and the search climbs to 2, and so on until a new candidate is
found or the topic entity is reached.
"""
import difflib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .base import Triple, norm_entity

Ranked = List[Tuple[Triple, float]]


@dataclass
class PathNode:
    key: str
    name: str
    id: str
    parent: Optional[str]
    ranked: Ranked = field(default_factory=list)
    dead: bool = False


class PathTracker:
    def __init__(self, topic_entities: List[Tuple[str, str]]):
        self.nodes: Dict[str, PathNode] = {}
        self.roots: List[str] = []
        # entities the model may search: topic entities and entities in shown triples
        self.known: Dict[str, Tuple[str, str, Optional[str]]] = {}
        for name, eid in topic_entities:
            key = norm_entity(name)
            self.roots.append(key)
            self.known[key] = (name, eid, None)
        self.current: Optional[str] = None

    # ----------------------------------------------------------- resolution
    def resolve(self, query: str) -> Optional[Tuple[str, str, Optional[str]]]:
        """Match a free-text entity mention to a known entity -> (name, id, parent_key)."""
        key = norm_entity(query)
        if not key:
            return None
        if key in self.known:
            return self.known[key]
        close = difflib.get_close_matches(key, list(self.known), n=1, cutoff=0.85)
        if close:
            return self.known[close[0]]
        if len(key) >= 4:
            subs = [k for k in self.known if len(k) >= 4 and (key in k or k in key)]
            if subs:
                return self.known[min(subs, key=lambda k: abs(len(k) - len(key)))]
        return None

    def key_of(self, query: str) -> Optional[str]:
        hit = self.resolve(query)
        return norm_entity(hit[0]) if hit else None

    def register_shown(self, node_key: str, triples: List[Triple]) -> None:
        for tr in triples:
            name, eid = tr.other_end(node_key)
            self.known.setdefault(norm_entity(name), (name, eid, node_key))

    # ------------------------------------------------------------ traversal
    def visit(self, name: str, eid: str, parent: Optional[str], ranked: Ranked) -> PathNode:
        key = norm_entity(name)
        node = self.nodes.get(key)
        if node is None:
            if parent == key:
                parent = None
            node = PathNode(key, name, eid, parent if parent in self.nodes else None, ranked)
            self.nodes[key] = node
        self.current = key
        return node

    def ancestors(self, key: Optional[str]) -> List[str]:
        out, seen = [], set()
        while key is not None and key in self.nodes and key not in seen:
            seen.add(key)
            out.append(key)
            key = self.nodes[key].parent
        return out

    def path_names(self, key: Optional[str] = None) -> List[str]:
        return [self.nodes[k].name for k in reversed(self.ancestors(key or self.current))]

    def candidates(self, node: PathNode, k: int) -> List[Triple]:
        """Top-k triples of `node` whose other end is neither a dead end nor an ancestor."""
        blocked = set(self.ancestors(node.key))
        out = []
        for tr, _ in node.ranked:
            other = norm_entity(tr.other_end(node.key)[0])
            if other in blocked or (other in self.nodes and self.nodes[other].dead):
                continue
            out.append(tr)
            if len(out) >= k:
                break
        return out

    def backtrack(self, dead_key: Optional[str], k: int) -> Tuple[Optional[PathNode], List[Triple], Optional[str]]:
        """Returns (node backtracked to, alternative triples, name of the dead-end entity)."""
        start = dead_key if dead_key in self.nodes else self.current
        if start is None:
            return None, [], None
        dead = self.nodes[start]
        dead.dead = True
        parent = dead.parent
        while parent is not None:
            node = self.nodes[parent]
            alts = self.candidates(node, k)
            if alts:
                self.current = node.key
                return node, alts, dead.name
            node.dead = True  # no alternative left here: climb one level higher
            parent = node.parent
        return None, [], dead.name

    def unexplored_roots(self) -> List[Tuple[str, str]]:
        return [self.known[r][:2] for r in self.roots if r not in self.nodes]
