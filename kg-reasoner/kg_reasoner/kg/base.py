"""Knowledge-graph interface used by the retrieval tool (paper Sec. 4.2)."""
import re
import string
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Triple:
    head: str
    relation: str
    tail: str
    head_id: str = ""
    tail_id: str = ""

    def render(self) -> str:
        return f"({self.head}, {self.relation}, {self.tail})"

    def other_end(self, entity_key: str) -> Tuple[str, str]:
        """(name, id) of the entity on the other side of `entity_key` (a normalized name)."""
        if norm_entity(self.head) == entity_key:
            return self.tail, self.tail_id or self.tail
        return self.head, self.head_id or self.head

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict) -> "Triple":
        return Triple(d["head"], d["relation"], d["tail"], d.get("head_id", ""), d.get("tail_id", ""))


def norm_entity(name: str) -> str:
    s = (name or "").lower().replace("_", " ")
    s = "".join(ch if ch not in string.punctuation else " " for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


class KnowledgeGraph(ABC):
    """One-hop access to a KG. `entity_id` is a KG id (e.g. a Freebase MID)."""

    name_ids = False  # True for graphs keyed by entity names instead of ids

    @abstractmethod
    def neighbors(self, entity_id: str, entity_name: Optional[str] = None, kg_style: str = "freebase") -> List[Triple]:
        """Outgoing (entity, r, o) and incoming (s, r, entity) triples (Sec. 4.2.1)."""

    def cached_neighbors(self, entity_id: str, kg_style: str = "freebase") -> Optional[List[Triple]]:
        """Neighbors if already fetched (no new lookup); None otherwise."""
        return None

    def resolve(self, name: str, kg_style: str = "freebase") -> Optional[str]:
        """Map an entity name produced by the LLM to a KG id (None if unknown)."""
        return name
