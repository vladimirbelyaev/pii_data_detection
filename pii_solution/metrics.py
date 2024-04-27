import numpy as np
from seqeval.metrics import recall_score, precision_score
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score


# from time import time

def compute_metrics(p, all_labels):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [all_labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [all_labels[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # print(true_predictions[:100])
    # print(true_labels[:100])

    recall = recall_score(true_labels, true_predictions)
    precision = precision_score(true_labels, true_predictions)
    f1_score = (1 + 5 * 5) * recall * precision / (5 * 5 * precision + recall)

    report = classification_report(true_labels, true_predictions, output_dict=True, zero_division=0)
    good_report = dict()
    for name, metrs in report.items():
        if name[0].islower():
            continue
        subrecall = metrs['recall']
        subprecision = metrs['precision']
        if subrecall == 0 and subprecision == 0:
            good_report[name + '_f5'] = np.nan
        else:
            good_report[name + '_f5'] = (1 + 5 * 5) * subrecall * subprecision / (5 * 5 * subprecision + subrecall)
    for name in all_labels:
        if name == 'O' or name[0] == 'I':
            continue
        good_report.setdefault(name[2:] + '_f5', np.nan)

    results = {
        'recall': recall,
        'precision': precision,
        'f1': f1_score,
        **good_report
    }
    return results
