from typing import Dict, Optional

import numpy as np
import transformers

TARGET = [
    'B-EMAIL', 'B-ID_NUM', 'B-NAME_STUDENT', 'B-PHONE_NUM',
    'B-STREET_ADDRESS', 'B-URL_PERSONAL', 'B-USERNAME', 'I-ID_NUM',
    'I-NAME_STUDENT', 'I-PHONE_NUM', 'I-STREET_ADDRESS', 'I-URL_PERSONAL'
]


def tokenize_train(
        example,
        tokenizer: transformers.PreTrainedTokenizer,
        label2id: Dict[str, int],
        max_length: int,
        truncation: bool = True,
        truncate_left_input_to: Optional[int] = None
):
    text = []

    # these are at the character level
    labels = []
    targets = []
    start_length = len(example['tokens'])
    if truncate_left_input_to is None:
        tokens_to_skip = None
    else:
        tokens_to_skip = max(start_length - truncate_left_input_to, 0)

    for idx, (t, l, ws) in enumerate(zip(example["tokens"], example["provided_labels"], example["trailing_whitespace"])):
        if tokens_to_skip is not None and idx < tokens_to_skip:
            continue
        
        text.append(t)
        labels.extend([l] * len(t))

        if l in TARGET:
            targets.append(1)
        else:
            targets.append(0)
        # if there is trailing whitespace
        if ws:
            text.append(" ")
            labels.append("O")

    tokenized = tokenizer("".join(text), return_offsets_mapping=True, truncation=truncation, max_length=max_length)

    target_num = sum(targets)
    labels = np.array(labels)

    text = "".join(text)
    token_labels = []

    for start_idx, end_idx in tokenized.offset_mapping:

        # CLS token
        if start_idx == 0 and end_idx == 0:
            token_labels.append(label2id["O"])
            continue

        # case when token starts with whitespace
        if text[start_idx].isspace():
            start_idx += 1
        # print(labels.shape, start_idx)
        token_labels.append(label2id[labels[start_idx]])

    length = len(tokenized.input_ids)

    return {
        **tokenized,
        "labels": token_labels,
        "length": length,
        "target_num": target_num,
        "group": 1 if target_num > 0 else 0
    }


def tokenize_eval(
        example,
        tokenizer: transformers.PreTrainedTokenizer,
        max_length: int,
        truncation: bool = True,
):
    # We be creatin' two empty lists, 'text' and 'token_map', to store our tokens and their respective maps.
    text = []
    token_map = []

    # We start the 'idx' at 0, it is used to keep track of the tokens.
    idx = 0

    # Now, we be loopin' through the tokens and their trailin' white spaces.
    for t, ws in zip(example["tokens"], example["trailing_whitespace"]):

        # We add the token 't' to the 'text' list.
        text.append(t)

        # We be extendin' the 'token_map' list by repeatin' the 'idx' as many times as the length of token 't'.
        token_map.extend([idx] * len(t))

        # If there be trailin' whitespace (ws), we add a space to 'text' and mark it with a '-1' in 'token_map'.
        if ws:
            text.append(" ")
            token_map.append(-1)

        # We increment 'idx' to keep track of the next token.
        idx += 1

    # Now, we tokenize the concatenated 'text' and return offsets mappings along with 'token_map'.
    tokenized = tokenizer("".join(text), return_offsets_mapping=True, truncation=truncation, max_length=max_length)

    # We return a dictionary containin' the tokenized data and the 'token_map'.
    return {
        **tokenized,
        "token_map": token_map,
    }
