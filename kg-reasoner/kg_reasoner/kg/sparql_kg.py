"""Freebase over SPARQL (Sec. 4.2.1). Requires a Virtuoso endpoint loaded with Freebase;
set --sparql_endpoint or SPARQL_ENDPOINT."""
import os
import threading
from collections import OrderedDict
from typing import List, Optional

from .base import KnowledgeGraph, Triple

FB = "http://rdf.freebase.com/ns/"

# Sec. 4.2.1: all outgoing predicates/objects, and incoming predicates where the entity is the object.
OUTGOING = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT ?predicate ?object ?name WHERE {
  ns:%s ?predicate ?object .
  OPTIONAL { ?object ns:type.object.name ?name . FILTER(langMatches(lang(?name), 'en')) }
} LIMIT %d"""

INCOMING = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT ?subject ?predicate ?name WHERE {
  ?subject ?predicate ns:%s .
  OPTIONAL { ?subject ns:type.object.name ?name . FILTER(langMatches(lang(?name), 'en')) }
} LIMIT %d"""

NAME = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT ?name WHERE { ns:%s ns:type.object.name ?name . FILTER(langMatches(lang(?name), 'en')) } LIMIT 1"""

RESOLVE = """PREFIX ns: <http://rdf.freebase.com/ns/>
SELECT ?entity WHERE { ?entity ns:type.object.name "%s"@en . } LIMIT 1"""


def skip_relation(rel: str) -> bool:
    return (rel in ("type.object.type", "type.object.name") or rel.startswith(("common.", "freebase.", "kg.", "type."))
            or "sameAs" in rel or rel.endswith((".key", ".id")))


class FreebaseSparqlKG(KnowledgeGraph):
    def __init__(self, endpoint: Optional[str] = None, limit: int = 500, cache_size: int = 100_000):
        from SPARQLWrapper import JSON, SPARQLWrapper

        self.endpoint = endpoint or os.environ.get("SPARQL_ENDPOINT", "http://localhost:8890/sparql")
        self.limit = limit
        self._wrapper, self._json = SPARQLWrapper, JSON
        self._names = {}
        self._cache: "OrderedDict[str, List[Triple]]" = OrderedDict()  # LRU of neighborhoods
        self._cache_size = cache_size
        self._lock = threading.Lock()

    def _query(self, q: str):
        sparql = self._wrapper(self.endpoint)
        sparql.setQuery(q)
        sparql.setReturnFormat(self._json)
        return sparql.query().convert()["results"]["bindings"]

    def name(self, mid: str) -> str:
        if mid not in self._names:
            rows = self._query(NAME % mid)
            self._names[mid] = rows[0]["name"]["value"] if rows else f"[{mid}]"  # unnamed (CVT) node
        return self._names[mid]

    def _node(self, binding, name_binding):
        value = binding["value"]
        if binding["type"] == "uri":
            mid = value.replace(FB, "")
            name = name_binding["value"] if name_binding else f"[{mid}]"
            self._names.setdefault(mid, name)
            return name, mid
        return value, ""  # literal

    def cached_neighbors(self, entity_id: str, kg_style: str = "freebase") -> Optional[List[Triple]]:
        with self._lock:
            return self._cache.get(entity_id)

    def neighbors(self, entity_id: str, entity_name: Optional[str] = None, kg_style: str = "freebase") -> List[Triple]:
        cached = self.cached_neighbors(entity_id)
        if cached is not None:
            return cached
        head = entity_name or self.name(entity_id)
        triples = []
        for row in self._query(OUTGOING % (entity_id, self.limit)):
            rel = row["predicate"]["value"].replace(FB, "")
            if skip_relation(rel):
                continue
            tail, tail_id = self._node(row["object"], row.get("name"))
            triples.append(Triple(head, rel, tail, entity_id, tail_id))
        for row in self._query(INCOMING % (entity_id, self.limit)):
            rel = row["predicate"]["value"].replace(FB, "")
            if skip_relation(rel):
                continue
            subj, subj_id = self._node(row["subject"], row.get("name"))
            triples.append(Triple(subj, rel, head, subj_id, entity_id))
        with self._lock:
            self._cache[entity_id] = triples
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return triples

    def resolve(self, name: str, kg_style: str = "freebase") -> Optional[str]:
        if name.startswith("[m.") and name.endswith("]"):
            return name[1:-1]
        rows = self._query(RESOLVE % name.replace('"', '\\"'))
        return rows[0]["entity"]["value"].replace(FB, "") if rows else None
