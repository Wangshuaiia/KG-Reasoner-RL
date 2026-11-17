"""Loading the eight benchmarks (paper Sec. 5.1 / Table 4) into one record format.

Unified record:
    {"id", "dataset", "kg": "freebase"|"wikidata", "task": "qa"|"slot_filling"|"fact_checking",
     "question": str, "answers": [str], "topic_entities": [{"id", "name"}], "hops": int|None}
"""
import json
import os
from typing import Dict, Iterable, List, Optional

DATASETS = {
    # name: (kg, task)
    "cwq": ("freebase", "qa"),
    "webqsp": ("freebase", "qa"),
    "webquestions": ("freebase", "qa"),
    "grailqa": ("freebase", "qa"),
    "qald": ("wikidata", "qa"),
    "trex": ("wikidata", "slot_filling"),
    "zeroshotre": ("wikidata", "slot_filling"),
    "creak": ("wikidata", "fact_checking"),
}

_FILE_ALIASES = {
    "cwq": "cwq", "webqsp": "webqsp", "webquestions": "webquestions", "graliqa": "grailqa",
    "grailqa": "grailqa", "qald_10-en": "qald", "qald": "qald", "t-rex": "trex", "trex": "trex",
    "zero_shot_re": "zeroshotre", "zeroshotre": "zeroshotre", "creak": "creak",
}


UNIFIED = "unified"


def detect_dataset(path: str, sample: Dict) -> str:
    if "topic_entities" in sample and "answers" in sample and "question" in sample:
        return UNIFIED  # already converted (e.g. scripts/build_rl_data.py output)
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    stem = stem.replace("_train", "").replace("_test", "").replace("-train", "").replace("-test", "")
    if stem in _FILE_ALIASES:
        return _FILE_ALIASES[stem]
    if "compositionality_type" in sample:
        return "cwq"
    if "Parses" in sample:
        return "webqsp"
    if "graph_query" in sample or "num_edge" in sample:
        return "grailqa"
    if "template_questions" in sample:
        return "zeroshotre"
    if "wikipedia_title" in sample:
        return "trex"
    if "sentence" in sample and "label" in sample:
        return "creak"
    if "url" in sample and "answers" in sample:
        return "webquestions"
    if "sparql" in sample and "qid_topic_entity" in sample:
        return "qald"
    raise ValueError(f"Cannot detect dataset for {path}")


def _as_list(answer) -> List[str]:
    if answer is None:
        return []
    if isinstance(answer, bool):
        return ["true" if answer else "false"]
    if isinstance(answer, dict):
        return [str(v) for v in answer.values()]
    if isinstance(answer, (list, tuple)):
        out = []
        for a in answer:
            if isinstance(a, dict):
                out.append(str(a.get("entity_name") or a.get("EntityName") or a.get("answer_argument")
                               or a.get("AnswerArgument") or ""))
            else:
                out.append(str(a))
        return [a for a in out if a]
    return [str(answer)]


def _entities(mapping) -> List[Dict[str, str]]:
    if not mapping:
        return []
    if isinstance(mapping, dict):
        return [{"id": str(k), "name": str(v)} for k, v in mapping.items()]
    return [{"id": str(e.get("id", e.get("name"))), "name": str(e.get("name"))} for e in mapping]


def normalize_record(raw: Dict, dataset: str, index: int) -> Dict:
    if dataset == UNIFIED:
        rec = dict(raw)
        rec.setdefault("id", f"{rec.get('dataset', 'sample')}-{index}")
        rec.setdefault("hops", None)
        rec.setdefault("kg", DATASETS.get(rec.get("dataset"), ("freebase", "qa"))[0])
        rec.setdefault("task", DATASETS.get(rec.get("dataset"), ("freebase", "qa"))[1])
        return rec
    kg, task = DATASETS[dataset]
    hops: Optional[int] = None
    if dataset == "cwq":
        rid, question, answers = raw["ID"], raw["question"], _as_list(raw.get("answer"))
        topics, hops = _entities(raw.get("topic_entity")), 2
    elif dataset == "webqsp":
        rid, question = raw["QuestionId"], raw["RawQuestion"]
        answers = []
        for parse in raw.get("Parses", []):
            answers += _as_list(parse.get("Answers"))
        answers = list(dict.fromkeys(answers))
        topics = _entities(raw.get("topic_entity"))
        chain = (raw.get("Parses") or [{}])[0].get("InferentialChain")
        hops = len(chain) if chain else None
    elif dataset == "grailqa":
        rid, question, answers = str(raw["qid"]), raw["question"], _as_list(raw.get("answer"))
        topics, hops = _entities(raw.get("topic_entity")), raw.get("num_edge")
    elif dataset == "webquestions":
        rid, question, answers = f"webquestions-{index}", raw["question"], _as_list(raw.get("answers"))
        topics = _entities(raw.get("topic_entity"))
    elif dataset == "qald":
        rid, question, answers = f"qald-{index}", raw["question"], _as_list(raw.get("answer"))
        topics = _entities(raw.get("qid_topic_entity"))
    elif dataset in ("trex", "zeroshotre"):
        rid, question = f"{dataset}-{index}", raw["input"]
        answers = list(dict.fromkeys(_as_list(raw.get("answer")) + _as_list(raw.get("alias"))))
        topics, hops = _entities(raw.get("qid_topic_entity") or raw.get("topic_entity_ids")), 1
    elif dataset == "creak":
        rid, question = raw.get("ex_id", f"creak-{index}"), raw["sentence"]
        answers = [str(raw["label"]).lower()]
        topics = _entities(raw.get("qid_topic_entity"))
    else:
        raise ValueError(dataset)
    return {"id": str(rid), "dataset": dataset, "kg": kg, "task": task, "question": question,
            "answers": answers, "topic_entities": topics, "hops": hops}


def _read_json_or_jsonl(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def load_records(path: str, limit: Optional[int] = None) -> List[Dict]:
    raw = _read_json_or_jsonl(path)
    if not raw:
        return []
    dataset = detect_dataset(path, raw[0])
    records = [normalize_record(r, dataset, i) for i, r in enumerate(raw)]
    records = [r for r in records if r["question"] and r["answers"]]
    return records[:limit] if limit else records


def write_jsonl(path: str, rows: Iterable[Dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
