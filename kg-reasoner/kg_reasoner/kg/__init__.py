"""KG retrieval tool: backends, entity ranking, backtracking (paper Sec. 4.2)."""
import os
from typing import Dict, Optional

from .base import KnowledgeGraph, Triple, norm_entity
from .tool import KGRetrievalTool, parse_search_query

BACKENDS = ("sparql", "remote")
RANKERS = ("gnn", "similarity")


def build_kg(backend: str = "sparql", sparql_endpoint: Optional[str] = None,
             server_url: Optional[str] = None) -> KnowledgeGraph:
    if backend == "sparql":
        from .sparql_kg import FreebaseSparqlKG

        return FreebaseSparqlKG(endpoint=sparql_endpoint)
    if backend == "remote":
        from .remote import RemoteKG

        return RemoteKG(server_url or os.environ.get("KG_SERVER_URL", "http://127.0.0.1:5003"))
    raise ValueError(f"Unknown KG backend {backend!r}; choose from {BACKENDS}")


def build_ranker(name: str, kg: KnowledgeGraph, gnn_ckpt: Optional[str] = None, encoder: Optional[str] = None,
                 device: str = "cpu", gnn_context: str = "fetch", top_k: int = 5):
    from .gnn import DEFAULT_ENCODER, TextEncoder, load_gnn
    from .rankers import ContextProvider, GNNRanker, SimilarityRanker

    if name == "similarity":
        return SimilarityRanker(TextEncoder(encoder or DEFAULT_ENCODER, device))
    if name == "gnn":
        if not gnn_ckpt or not os.path.exists(gnn_ckpt):
            raise FileNotFoundError(
                f"GNN checkpoint {gnn_ckpt!r} not found. Train one with scripts/train_gnn.py, "
                "or use --kg_ranker similarity.")
        model, enc_name = load_gnn(gnn_ckpt, device)
        return GNNRanker(model, TextEncoder(encoder or enc_name, device), ContextProvider(kg, gnn_context))
    raise ValueError(f"Unknown ranker {name!r}; choose from {RANKERS}")


class ToolFactory:
    """Creates one KGRetrievalTool per episode, sharing the KG, ranker and ranking cache."""

    def __init__(self, kg: KnowledgeGraph, ranker, top_k: int = 5):
        self.kg, self.ranker, self.top_k = kg, ranker, top_k
        self.rank_cache: Dict = {}

    def __call__(self, record: Dict) -> KGRetrievalTool:
        if len(self.rank_cache) > 100_000:
            self.rank_cache.clear()
        return KGRetrievalTool(self.kg, self.ranker, record["question"], record.get("topic_entities") or [],
                               kg_style=record.get("kg", "freebase"), top_k=self.top_k, rank_cache=self.rank_cache)


def build_tool_factory(backend: str = "sparql", ranker: str = "gnn", gnn_ckpt: Optional[str] = None,
                       server_url: Optional[str] = None, top_k: int = 5, gnn_context: Optional[str] = None,
                       device: str = "cpu", sparql_endpoint: Optional[str] = None,
                       encoder: Optional[str] = None) -> ToolFactory:
    kg = build_kg(backend, sparql_endpoint=sparql_endpoint, server_url=server_url)
    if backend == "remote":
        from .remote import RemoteRanker

        rk = RemoteRanker(server_url or os.environ.get("KG_SERVER_URL", "http://127.0.0.1:5003"))
    else:
        rk = build_ranker(ranker, kg, gnn_ckpt=gnn_ckpt, encoder=encoder, device=device,
                          gnn_context=gnn_context or "fetch", top_k=top_k)
    return ToolFactory(kg, rk, top_k=top_k)


def add_kg_args(parser) -> None:
    g = parser.add_argument_group("KG retrieval tool")
    g.add_argument("--kg_backend", default="sparql", choices=BACKENDS,
                   help="sparql: Freebase SPARQL endpoint; remote: shared kg_reasoner.kg.server")
    g.add_argument("--kg_server_url", default=None)
    g.add_argument("--sparql_endpoint", default=None, help="default: $SPARQL_ENDPOINT or http://localhost:8890/sparql")
    g.add_argument("--kg_ranker", default="gnn", choices=RANKERS)
    g.add_argument("--gnn_ckpt", default=None)
    g.add_argument("--kg_topk", type=int, default=5, help="triples returned per search")
    g.add_argument("--gnn_context", default=None, choices=[None, "none", "cached", "fetch"])


def tool_factory_from_args(args, device: str = "cpu") -> ToolFactory:
    return build_tool_factory(backend=args.kg_backend, ranker=args.kg_ranker, gnn_ckpt=args.gnn_ckpt,
                              server_url=args.kg_server_url, top_k=args.kg_topk, gnn_context=args.gnn_context,
                              device=device, sparql_endpoint=args.sparql_endpoint)
