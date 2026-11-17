"""Shared helpers for SFT warm-up, hard-case selection (Eq. 10) and local GRPO."""
from typing import List, Tuple

import torch

from .prompts import split_response_segments

IGNORE = -100


def pick_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_sft_example(tokenizer, prompt: str, response: str, max_len: int = 4096,
                      add_eos: bool = True) -> Tuple[List[int], List[int]]:
    """Token ids and labels; the prompt and tool-inserted <triples> blocks get IGNORE labels.
    Segments are tokenized separately, exactly as the rollout builds sequences."""
    ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    labels = [IGNORE] * len(ids)
    for text, generated in split_response_segments(response):
        seg = tokenizer(text, add_special_tokens=False)["input_ids"]
        ids += seg
        labels += seg if generated else [IGNORE] * len(seg)
    if add_eos and tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
        labels.append(tokenizer.eos_token_id)
    return ids[:max_len], labels[:max_len]


def token_logprobs(model, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
    """log p(x_t | x_<t) for t = 1..T-1, shape [B, T-1]."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1].float()
    return torch.log_softmax(logits, dim=-1).gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)


def sequence_nll(model, ids: List[int], labels: List[int], device: str) -> float:
    """Eq. 10: L_sft = -1/T * sum_t log P(a_t | a_<t, s) over supervised tokens."""
    x = torch.tensor([ids], device=device)
    y = torch.tensor([labels], device=device)[:, 1:]
    with torch.no_grad():
        lp = token_logprobs(model, x)
    mask = (y != IGNORE).float()
    return float(-(lp * mask).sum() / mask.sum().clamp(min=1))


def sft_loss(model, batch: List[Tuple[List[int], List[int]]], pad_id: int, device: str) -> torch.Tensor:
    n = max(len(ids) for ids, _ in batch)
    x = torch.full((len(batch), n), pad_id, dtype=torch.long)
    y = torch.full((len(batch), n), IGNORE, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, (ids, labels) in enumerate(batch):
        x[i, :len(ids)] = torch.tensor(ids)
        y[i, :len(labels)] = torch.tensor(labels)
        att[i, :len(ids)] = 1
    x, y, att = x.to(device), y.to(device), att.to(device)
    lp = token_logprobs(model, x, att)
    mask = (y[:, 1:] != IGNORE).float()
    return -(lp * mask).sum() / mask.sum().clamp(min=1)
