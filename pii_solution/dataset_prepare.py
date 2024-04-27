import json
from itertools import chain
from typing import Dict, List, Tuple

from datasets import Dataset


def create_dataset(data: List[Dict]) -> Dataset:
    return Dataset.from_dict({
        "full_text": [x["full_text"] for x in data],
        "document": [x["document"] for x in data],
        "tokens": [x["tokens"] for x in data],
        "trailing_whitespace": [x["trailing_whitespace"] for x in data],
        "provided_labels": [x["labels"] for x in data],
    })


def load_data(path: str, offset: int = 0) -> List[Dict]:
    with open(path) as f:
        data = json.load(f)
    if offset > 0:
        for idx, doc in enumerate(data):
            doc['document'] = idx + offset
    return data


def create_labels(data: List[Dict]) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    all_labels = sorted(list(set(chain(*[x["labels"] for x in data]))))
    label2id = {l: i for i, l in enumerate(all_labels)}
    id2label = {v: k for k, v in label2id.items()}
    return all_labels, label2id, id2label


def renumber_dataset(row: Dict, first_index: int) -> Dataset:
    return {**row, 'document': row['document'] + first_index}