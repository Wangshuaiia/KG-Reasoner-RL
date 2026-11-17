"""GNN-based entity ranking (paper Sec. 4.2.2, Eq. 1-4).

For a candidate entity h with one-hop neighbors (t_i via relation r_i) and question q:
    s_i     = a(W q ; W (t_i ⊙ r_i))                       (Eq. 1, GAT-style attention)
    alpha_i = softmax(s_i)
    h_hat   = sigma(W_h [h || sum_i alpha_i t_i])
    y_hat   = Softmax(W_2 sigma(W_1 h_hat + b_1) + b_2)     (Eq. 2)
    score   = y_hat[1]                                       (Eq. 3)
trained with cross-entropy on relevant / irrelevant labels (Eq. 4).
Text of questions, entities and relations is embedded by a frozen sentence encoder.
"""
import re
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


def relation_text(relation: str) -> str:
    return " ".join(re.split(r"[./_:\s]+", relation)).strip()


class TextEncoder:
    """Frozen mean-pooled transformer encoder with an embedding cache."""

    def __init__(self, name: str = DEFAULT_ENCODER, device: str = "cpu", max_cache: int = 200_000):
        from transformers import AutoModel, AutoTokenizer

        self.name, self.device = name, device
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(device).eval()
        self.dim = self.model.config.hidden_size
        self._cache: Dict[str, torch.Tensor] = {}
        self.max_cache = max_cache

    @torch.no_grad()
    def encode(self, texts: Sequence[str], batch_size: int = 128) -> torch.Tensor:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        for i in range(0, len(missing), batch_size):
            chunk = missing[i:i + batch_size]
            enc = self.tokenizer(chunk, padding=True, truncation=True, max_length=64, return_tensors="pt").to(self.device)
            out = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = F.normalize((out * mask).sum(1) / mask.sum(1).clamp(min=1e-6), dim=-1).cpu()
            if len(self._cache) + len(chunk) > self.max_cache:
                self._cache.clear()
            self._cache.update(zip(chunk, emb))
        if not texts:
            return torch.zeros(0, self.dim)
        return torch.stack([self._cache[t] for t in texts])


class GNNEntityScorer(nn.Module):
    def __init__(self, dim: int, hidden: int = 128, mlp_hidden: int = 64):
        super().__init__()
        self.W = nn.Linear(dim, hidden, bias=False)       # shared W in Eq. 1
        self.a = nn.Linear(2 * hidden, 1, bias=False)     # attention function a(.;.)
        self.W_h = nn.Linear(2 * dim, hidden)             # W_h in Eq. 1
        self.mlp = nn.Sequential(nn.Linear(hidden, mlp_hidden), nn.ReLU(), nn.Linear(mlp_hidden, 2))  # Eq. 2
        self.config = {"dim": dim, "hidden": hidden, "mlp_hidden": mlp_hidden}

    def forward(self, q, h, t, r, mask):
        """q, h: [B, d]; t, r: [B, N, d]; mask: [B, N] (bool). Returns logits [B, 2]."""
        n = t.size(1)
        wq = self.W(q).unsqueeze(1).expand(-1, n, -1)
        wtr = self.W(t * r)
        s = F.leaky_relu(self.a(torch.cat([wq, wtr], dim=-1)).squeeze(-1), 0.2)
        s = s.masked_fill(~mask, -1e9)
        alpha = torch.softmax(s, dim=-1) * mask.float()  # entities without neighbors aggregate nothing
        agg = (alpha.unsqueeze(-1) * t).sum(dim=1)
        h_hat = torch.relu(self.W_h(torch.cat([h, agg], dim=-1)))
        return self.mlp(h_hat)

    @torch.no_grad()
    def score(self, q, h, t, r, mask) -> torch.Tensor:
        return torch.softmax(self(q, h, t, r, mask), dim=-1)[:, 1]  # Eq. 3


Candidate = Tuple[str, List[Tuple[str, str]]]  # (entity name, [(relation, neighbor name), ...])


def build_batch(encoder: TextEncoder, question: str, candidates: List[Candidate], max_neighbors: int = 32):
    """Embed a question and its candidate entities with their neighbors into GNN inputs."""
    neigh = [c[1][:max_neighbors] for c in candidates]
    n = max(1, max((len(x) for x in neigh), default=1))
    q = encoder.encode([question]).expand(len(candidates), -1)
    h = encoder.encode([c[0] for c in candidates])
    rel_texts = [relation_text(r) for x in neigh for r, _ in x]
    tail_texts = [t for x in neigh for _, t in x]
    rel_emb, tail_emb = encoder.encode(rel_texts), encoder.encode(tail_texts)
    d = encoder.dim
    t = torch.zeros(len(candidates), n, d)
    r = torch.zeros(len(candidates), n, d)
    mask = torch.zeros(len(candidates), n, dtype=torch.bool)
    k = 0
    for i, x in enumerate(neigh):
        if x:
            t[i, :len(x)] = tail_emb[k:k + len(x)]
            r[i, :len(x)] = rel_emb[k:k + len(x)]
            mask[i, :len(x)] = True
            k += len(x)
    return q, h, t, r, mask


def save_gnn(path: str, model: GNNEntityScorer, encoder_name: str) -> None:
    torch.save({"state_dict": model.state_dict(), "config": model.config, "encoder": encoder_name}, path)


def load_gnn(path: str, device: str = "cpu") -> Tuple[GNNEntityScorer, str]:
    ckpt = torch.load(path, map_location=device)
    model = GNNEntityScorer(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt.get("encoder", DEFAULT_ENCODER)
