import gc
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from enum import Enum
from scipy.special import softmax
from seqeval.metrics import precision_score, recall_score
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForTokenClassification, \
    DataCollatorForTokenClassification
from spacy.lang.en import English

nlp = English()
email_regex = re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+')
phone_num_regex = re.compile(r"(\(\d{3}\)\d{3}\-\d{4}\w*|\d{3}\.\d{3}\.\d{4})\s")


def find_span(target: list[str], document: list[str]) -> list[list[int]]:
    idx = 0
    spans = []
    span = []

    for i, token in enumerate(document):
        if token != target[idx]:
            idx = 0
            span = []
            continue
        span.append(i)
        
        idx += 1
        if idx == len(target):
            spans.append(span)
            span = []
            idx = 0
            continue
    
    return spans


def find_emails_phones(data):
    emails = []
    phone_nums = []

    for _data in data:
        # email
        for token_idx, token in enumerate(_data["tokens"]):
            if re.fullmatch(email_regex, token) is not None:
                emails.append(
                    {"document": _data["document"], "token": token_idx, "label": "B-EMAIL", "token_str": token, 'score': 1.0}
                )
        # phone number
        matches = phone_num_regex.findall(_data["full_text"])
        if not matches:
            continue
            
        for match in matches:
            target = [t.text for t in nlp.tokenizer(match)]
            matched_spans = find_span(target, _data["tokens"])
            
        for matched_span in matched_spans:
            for intermediate, token_idx in enumerate(matched_span):
                prefix = "I" if intermediate else "B"
                phone_nums.append(
                    {"document": _data["document"], "token": token_idx, "label": f"{prefix}-PHONE_NUM", "token_str": _data["tokens"][token_idx], 'score': 1.0}
                )
    return emails, phone_nums


def make_prediction_one(model, tokenizer, collator, ds):
    args = TrainingArguments(
        ".",
        per_device_eval_batch_size=1,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    predictions = trainer.predict(ds).predictions
    weighted_predictions = softmax(predictions, axis = -1)
    del model, trainer
    torch.cuda.empty_cache()
    gc.collect()
    return weighted_predictions


def make_prediction_many(model2weights, ds):
    all_predictions = []
    total_weight = 0
    for model_path, weight in model2weights.items():
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of = 16)
        predictions = make_prediction_one(model, tokenizer, collator, ds)
        all_predictions.append(predictions * weight)
        total_weight += weight
    weighted_average_predictions = np.sum(all_predictions, axis=0) / total_weight
    return weighted_average_predictions


class PredictionNormalizationMode(Enum):
    NO_REWRITE = 0
    REWRITE_STARTING_I = 1
    REWRITE_BOTH = 2



def create_df_submit(preds_final, ds_test, id2label: Dict[int, str], rewrite_mode: PredictionNormalizationMode, O_preds: torch.Tensor, ignored_labels: Optional[List] = None):
    document, token, label, token_str, probs_pred = [], [], [], [], []
    ignored_labels = ignored_labels or []
    # For each prediction, token mapping, offsets, tokens, and document in the dataset
    for idx, (p, token_map, offsets, tokens, doc) in enumerate(zip(
        preds_final, ds_test["token_map"], 
        ds_test["offset_mapping"],
        ds_test["tokens"], 
        ds_test["document"],
    )):
        triplets = list()
        probs = O_preds[idx, :]
        # print(f'p {len(p)} tkmap {len(token_map)} offs {len(offsets)} tok {len(tokens)}')
        # Iterate through each token prediction and its corresponding offsets
        for token_pred, (start_idx, end_idx), prob in zip(p, offsets, probs):
            label_pred = id2label[token_pred]  # Predicted label from token
            prob_predicted = probs[token_pred]

            # If start and end indices sum to zero, continue to the next iteration
            if start_idx + end_idx == 0:
                # print('start_idx')
                continue

            # If the token mapping at the start index is -1, increment start index
            if token_map[start_idx] == -1:
                # print('shift index', start_idx)
                start_idx += 1

            # Ignore leading whitespace tokens ("\n\n")
            while start_idx < len(token_map) and tokens[token_map[start_idx]].isspace():
                # print('ignore whitespace')
                start_idx += 1

            # If start index exceeds the length of token mapping, break the loop
            if start_idx >= len(token_map):
                break

            token_id = token_map[start_idx]  # Token ID at start index

            # Ignore "O" predictions and whitespace tokens
            if (label_pred != "O" and (label_pred not in ignored_labels)) and token_id != -1:
                # print(doc, token_id, label_pred, tokens[token_id], start_idx)
                prev_doc = document[-1] if document else None
                prev_label = label[-1] if label else None
                prev_token = token[-1] if token else None
                next_token_different_entity = prev_doc != doc or prev_token != (token_id - 1) or label_pred[2:] != prev_label[2:]
                if rewrite_mode == PredictionNormalizationMode.REWRITE_STARTING_I:
                    if next_token_different_entity and label_pred[:2] == 'I-':
                        label_pred = label_pred.replace('I-', 'B-')
                        #print('Replace I -> B', doc, token_id)
                elif rewrite_mode == PredictionNormalizationMode.REWRITE_BOTH:
                    if next_token_different_entity and label_pred[:2] == 'I-':
                        label_pred = label_pred.replace('I-', 'B-')
                        #print('Replace I -> B', doc, token_id)
                    elif not next_token_different_entity and label_pred[:2] == 'B-':
                        label_pred = label_pred.replace('B-', 'I-')
                        #print('Replace B -> I', document, token_id)
                triplet = (
                    token_id,
                    tokens[token_id],
                    # label_pred,
                )  # Form a triplet

                # If the triplet is not in the list of triplets, add it
                if triplet not in triplets:
                    # if some_change:
                    #     print('change', doc, token_id, label_pred, tokens[token_id])
                    # print('add', triplet)
                    document.append(doc)
                    token.append(token_id)
                    label.append(label_pred)
                    token_str.append(tokens[token_id])
                    triplets.append(triplet)
                    probs_pred.append(prob_predicted)

    df_fin = pd.DataFrame({
        "document": document,
        "token": token,
        "label": label,
        "token_str": token_str,
        'score': probs_pred
    }).sort_values('document')
    return df_fin


def calibrate(ds_test, real_data, id2label, O_preds, preds_without_O, preds, step, normalization_mode: PredictionNormalizationMode, regex_phone_email = False):
    if isinstance(step, float):
        rng = np.arange(0, 1, step)
    else:
        rng = step
    best_f5 = -1
    best_data = None
    ignored_labels = [] if not regex_phone_email else ["B-EMAIL", "B-PHONE_NUM", "I-PHONE_NUM"]
    tables = dict()
    print('===========ALL REAL PREDS===========')
    for threshold in rng:
        preds_final = np.where(O_preds < threshold, preds_without_O, preds)
        df_fin = create_df_submit(preds_final, ds_test, id2label, normalization_mode, O_preds, ignored_labels)
        if regex_phone_email:
            emails_data, phones_data = find_emails_phones(ds_test)
            regex_data = pd.DataFrame(emails_data + phones_data)
            df_fin = pd.concat([df_fin, regex_data], axis=0, ignore_index=True)
        #print(df_fin.shape)
        df_all = pd.merge(real_data, df_fin, how='outer', on=['document', 'token'], suffixes=['_real', '_pred']).fillna('O')
        #print(df_all.shape)
        tables[f'{threshold:.2f}'] = df_all
        true_labels = df_all.groupby('document')['label_real'].apply(list)
        pred_labels = df_all.groupby('document')['label_pred'].apply(list)
        subprecision = precision_score(true_labels, pred_labels)
        subrecall = recall_score(true_labels, pred_labels)
        f5 = (1 + 5*5) * subrecall * subprecision / (5*5*subprecision + subrecall)
        if f5 > best_f5:
            best_data = {'best_f5': f5, 'best_precision': subprecision, 'best_recall': subrecall, 'best_threshold': threshold}
            best_f5 = f5
        print(f'{threshold:.2f}: prec {subprecision:.4f}, rec {subrecall:.4f}, f5 {f5:.4f}')
    print('===========ALL PREDS W/O O===========')
    return best_data, tables
