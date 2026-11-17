"""Shared KG server: one process owns the KG connection, its neighbor cache and the entity ranker.

    python -m kg_reasoner.kg.server --kg_backend sparql --sparql_endpoint http://localhost:8890/sparql \
        --kg_ranker gnn --gnn_ckpt ckpt/gnn.pt --port 5003

Rollout workers then use --kg_backend remote --kg_server_url http://<host>:5003.
"""
import argparse
import logging
import threading

from . import add_kg_args, build_kg, build_ranker
from .base import Triple


def create_app(kg, ranker):
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(title="KG-Reasoner KG server")
    rank_lock = threading.Lock()

    class NeighborsReq(BaseModel):
        entity_id: str
        entity_name: str | None = None
        kg_style: str = "freebase"

    class CachedReq(BaseModel):
        entity_id: str
        kg_style: str = "freebase"

    class ResolveReq(BaseModel):
        name: str
        kg_style: str = "freebase"

    class RankReq(BaseModel):
        question: str
        entity_name: str
        entity_id: str
        kg_style: str = "freebase"
        triples: list[dict]

    @app.get("/info")
    def info():
        return {"name_ids": bool(getattr(kg, "name_ids", False))}

    @app.post("/neighbors")
    def neighbors(req: NeighborsReq):
        return {"triples": [t.to_dict() for t in kg.neighbors(req.entity_id, req.entity_name, req.kg_style)]}

    @app.post("/cached_neighbors")
    def cached(req: CachedReq):
        out = kg.cached_neighbors(req.entity_id, req.kg_style)
        return {"triples": None if out is None else [t.to_dict() for t in out]}

    @app.post("/resolve")
    def resolve(req: ResolveReq):
        return {"id": kg.resolve(req.name, req.kg_style)}

    @app.post("/rank")
    def rank(req: RankReq):
        triples = [Triple.from_dict(t) for t in req.triples]
        with rank_lock:
            ranked = ranker.rank(req.question, req.entity_name, req.entity_id, req.kg_style, triples)
        return {"triples": [t.to_dict() for t, _ in ranked], "scores": [float(s) for _, s in ranked]}

    return app


def main():
    parser = argparse.ArgumentParser()
    add_kg_args(parser)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5003)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.kg_backend == "remote":
        parser.error("the server needs a concrete backend (sparql)")
    logging.basicConfig(level=logging.INFO)
    kg = build_kg(args.kg_backend, sparql_endpoint=args.sparql_endpoint)
    ctx = args.gnn_context or "fetch"
    ranker = build_ranker(args.kg_ranker, kg, gnn_ckpt=args.gnn_ckpt, device=args.device, gnn_context=ctx,
                          top_k=args.kg_topk)
    import uvicorn

    uvicorn.run(create_app(kg, ranker), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
