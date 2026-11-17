import json

from torch.utils.data import Dataset
from tqdm import tqdm

from kg_reasoner.prompts import build_prompt

META_SPLIT = "<|idx_prompt_split|>"
META_KEYS = ("id", "dataset", "question", "topic_entities", "kg", "task")


def preprocess_data(data, input_template=None, input_key="question", apply_chat_template=None) -> str:
    """KG-Reasoner prompt (paper Appendix C) prefixed with the record metadata the KG tool needs.

    Records are in the unified format of kg_reasoner.data (scripts/build_rl_data.py output).
    Returns "<json meta><|idx_prompt_split|><prompt>".
    """
    if apply_chat_template:
        raise ValueError("KG-Reasoner uses the plain-text prompt of the paper; do not pass --apply_chat_template")
    record = {k: data.get(k) for k in META_KEYS}
    record["question"] = data[input_key]
    record["topic_entities"] = [dict(te) for te in (record.get("topic_entities") or [])]
    record["kg"] = record.get("kg") or "freebase"
    record["task"] = record.get("task") or "qa"
    return json.dumps(record, ensure_ascii=False) + META_SPLIT + build_prompt(record)


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None) or "question"
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        self.prompts = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            self.prompts.append(preprocess_data(data, input_template, input_key, apply_chat_template))

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]
