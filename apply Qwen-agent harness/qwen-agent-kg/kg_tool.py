"""Qwen-Agent tool `kg_search`: one-hop knowledge-graph lookup, bound to one question."""
import json5
from qwen_agent.tools.base import BaseTool, register_tool

from kg_interface import KnowledgeGraph


@register_tool("kg_search")
class KGSearch(BaseTool):
    description = ("Look up an entity in a knowledge graph. Returns facts as (head, relation, tail) triples that are "
                   "directly attached to the entity, most relevant first. Look up one entity per call; to follow a "
                   "multi-hop path, look up an entity that appeared in an earlier result.")
    parameters = {
        "type": "object",
        "properties": {
            "entity": {"type": "string", "description": "Name of the entity to look up, e.g. 'San Francisco Giants'"},
            "relation": {"type": "string", "description": "Optional: the kind of fact needed, e.g. 'championships'"},
        },
        "required": ["entity"],
    }

    def __init__(self, kg: KnowledgeGraph, question: str, top_k: int = 10):
        super().__init__()
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.kg, self.question, self.top_k = kg, question, top_k
        self.trace = []

    def call(self, params, **kwargs) -> str:
        args = params if isinstance(params, dict) else json5.loads(params)
        entity = str(args.get("entity") or next((v for v in args.values() if isinstance(v, str)), "")).strip()
        relation = str(args.get("relation") or "").strip()
        if not entity:
            return "Error: give the name of an entity to look up, e.g. {\"entity\": \"Utah\"}."
        canonical, triples = self.kg.lookup(entity, self.question, relation, self.top_k)
        self.trace.append({"entity": entity, "relation": relation, "resolved": canonical, "triples": triples})
        if not triples:
            return (f"No facts found for '{entity}'. Check the name, or look up another entity "
                    "from the question or from earlier results.")
        return "\n".join(f"({h}, {r}, {t})" for h, r, t in triples)
