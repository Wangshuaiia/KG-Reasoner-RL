"""Integration boundary for a real knowledge graph; no backend is bundled here."""
from abc import ABC, abstractmethod
from importlib import import_module
from typing import List, Optional, Tuple

Triple = Tuple[str, str, str]
LookupResult = Tuple[str, List[Triple]]


class KnowledgeGraph(ABC):
    """Resolve entity names and retrieve evidence from an existing knowledge graph."""

    @abstractmethod
    def lookup(self, entity: str, question: str = "", relation: str = "", k: int = 10) -> LookupResult:
        """Return (canonical entity name, up to k one-hop (head, relation, tail) triples).

        Every triple must come from the connected graph and touch the resolved entity.
        Use question and relation only to filter or rank graph evidence. Return
        ("", []) for an unknown entity and (canonical_name, []) for a known entity
        with no matching facts. Raise an exception for connection or query failures;
        an unavailable graph must not be reported as an empty successful lookup.
        """
        raise NotImplementedError


def load_knowledge_graph(backend: str, config: Optional[dict] = None) -> KnowledgeGraph:
    """Instantiate an external MODULE:CLASS implementation with configuration kwargs."""
    module_name, separator, class_name = backend.partition(":")
    if not separator or not module_name or not class_name or ":" in class_name:
        raise ValueError("KG backend must have the form MODULE:CLASS implementing kg_interface.KnowledgeGraph")
    if config is not None and not isinstance(config, dict):
        raise TypeError("KG configuration must be a JSON object of constructor keyword arguments")
    backend_class = getattr(import_module(module_name), class_name)
    if not isinstance(backend_class, type) or not issubclass(backend_class, KnowledgeGraph):
        raise TypeError("KG backend must be a subclass of kg_interface.KnowledgeGraph")
    return backend_class(**(config or {}))
