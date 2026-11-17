"""Special tags and prompts (paper Sec. 4.1 and Appendix C)."""
import re
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Special tags. Retrieved triples are wrapped in <triples> ... </triples> everywhere
# (SFT data, RL rollouts, reward and evaluation), including the warm-up prompt.
# ---------------------------------------------------------------------------
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
SEARCH_OPEN, SEARCH_CLOSE = "<search>", "</search>"
TRIPLES_OPEN, TRIPLES_CLOSE = "<triples>", "</triples>"
ANSWER_OPEN, ANSWER_CLOSE = "<answer>", "</answer>"

# Generation stops when the model closes a search call (tool is invoked) or the answer.
STOP_STRINGS = [SEARCH_CLOSE, ANSWER_CLOSE]

# Backtracking call (Sec. 4.2.3): the retrieval tool takes a Backtrack flag. The model
# triggers it with "<search> Backtrack=True </search>" (optionally naming the dead-end
# entity, e.g. "<search> Lou Seal, Backtrack=True </search>").
BACKTRACK_CALL = "Backtrack=True"

ASSISTANT_PREFIX = "Assistant: " + THINK_OPEN  # Algorithm 1: "Start with <think>"

# ---------------------------------------------------------------------------
# Appendix C: "System Prompt for LLMs" (used for SFT targets, RL and inference),
# with one sentence describing the Backtrack call of the retrieval tool (Sec. 4.2.3).
# ---------------------------------------------------------------------------
_TASK_LINE = {
    "qa": "The User asks a question, and the Assistant solves it.",
    "slot_filling": (
        "The User gives a sentence of the form 'subject [SEP] relation' with a missing object, "
        "and the Assistant fills the slot with the correct entity."
    ),
    "fact_checking": (
        "The User gives a claim, and the Assistant verifies whether it is true or false. "
        "The final answer must be either 'true' or 'false'."
    ),
}

SYSTEM_PROMPT_BODY = (
    "The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer.\n"
    "The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> "
    'tags, respectively, i.e., "<think> reasoning process here </think>\n<answer> final answer here </answer>".\n'
    "During the thinking process, the Assistant can perform searching for uncertain knowledge from a knowledge graph "
    'if necessary with the format of "<search> search query (only one entity) </search>". '
    "The entity must be involved in the searched triples or topic entities.\n"
    'Then, the system will provide the Assistant with helpful information with the format of "<triples> ...search results... </triples>".\n'
    "If the searched triples show that the current entity cannot support further reasoning, the Assistant can backtrack "
    'with "<search> Backtrack=True </search>", and the system will return alternative triples from the previous entity on the reasoning path.\n'
)

# ---------------------------------------------------------------------------
# Appendix C: "Prompt for Constructing Warm Up Data" (used with an open-weight
# teacher model, see scripts/build_warmup_data.py --teacher hf:/vllm:).
# ---------------------------------------------------------------------------
WARMUP_PROMPT_TEMPLATE = """You are a knowledgeable assistant that answers user questions through explicit reasoning and, when necessary, knowledge graph searches.
Instructions:
1. The assistant first thinks through the reasoning process before providing the final answer.
2. The reasoning process must be enclosed in `<think> ... </think>` tags.
3. The final answer must be enclosed in `<answer> ... </answer>` tags.
4. If the assistant lacks specific knowledge during reasoning, it may query a knowledge graph by issuing a search request using the format:
`<search> [ENTITY] </search>`
* Only one entity is allowed per search.
* The entity must be related to the question (e.g., a topic entity or one present in previously retrieved triples).
5. The system will respond with relevant knowledge in the format:
`<triples> (subject, predicate, object) </triples>`
6. The assistant must incorporate any retrieved triples into its reasoning process.
Example:
User: What timezone is Utah in?
Topic Entity: Utah
Assistant:
<think>
I am unsure about the timezone of Utah.
I will perform a search to retrieve relevant information.
<search>Utah</search>
<triples> (Utah, timeZone, Mountain Time Zone) </triples>
Based on the search results, I found that Utah is in the Mountain Time Zone.
</think>
<answer>Mountain Time Zone</answer>
Now begin answering the question.
User: {question}
Topic Entity: {topic_entity}
"""


def topic_entity_str(record: Dict) -> str:
    names = [te["name"] for te in record.get("topic_entities") or [] if te.get("name")]
    return "; ".join(dict.fromkeys(names)) if names else "None"


def build_prompt(record: Dict) -> str:
    """Inference/RL prompt (Appendix C system prompt), ending with "Assistant: <think>"."""
    task_line = _TASK_LINE.get(record.get("task", "qa"), _TASK_LINE["qa"])
    return (
        f"{task_line}\n{SYSTEM_PROMPT_BODY}"
        f"User: {record['question']}\n  Topic Entity: {topic_entity_str(record)}\n{ASSISTANT_PREFIX}"
    )


def build_warmup_prompt(record: Dict) -> str:
    return (
        WARMUP_PROMPT_TEMPLATE.format(question=record["question"], topic_entity=topic_entity_str(record))
        + ASSISTANT_PREFIX
    )


def format_observation(tool_output: str) -> str:
    """Wrap tool output exactly as it is inserted after </search> (shared by rollout and SFT data)."""
    return f"\n{TRIPLES_OPEN}\n{tool_output.strip()}\n{TRIPLES_CLOSE}\n"


_QUESTION_RE = re.compile(r"\nUser: (.*?)\n\s*Topic Entity: (.*?)\nAssistant:", re.S)


def split_query(full_text: str) -> Tuple[Optional[str], Optional[str], str]:
    """Split a decoded prompt+response into (question, topic_entity_str, response).
    Uses the first match: the RL prompt has exactly one, and a policy may imitate it in its response."""
    m = _QUESTION_RE.search(full_text)
    if m is None:
        return None, None, full_text
    response = full_text[m.end():]
    if response.startswith(" " + THINK_OPEN):
        response = response[len(THINK_OPEN) + 1:]
    elif response.startswith(THINK_OPEN):
        response = response[len(THINK_OPEN):]
    return m.group(1).strip(), m.group(2).strip(), response


def last_search_query(generated_text: str) -> str:
    """Text inside the last <search> ... </search> of a generated segment."""
    if SEARCH_OPEN not in generated_text:
        return ""
    return generated_text.rsplit(SEARCH_OPEN, 1)[-1].split(SEARCH_CLOSE, 1)[0].strip()


def split_response_segments(response: str) -> List[Tuple[str, bool]]:
    """Split a response into (text, generated_by_model) segments; <triples> blocks are tool output."""
    pattern = re.compile(r"(\n?" + re.escape(TRIPLES_OPEN) + r".*?" + re.escape(TRIPLES_CLOSE) + r"\n?)", re.S)
    segments = []
    for i, part in enumerate(pattern.split(response)):
        if part:
            segments.append((part, i % 2 == 0))
    return segments
