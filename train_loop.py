import os
import shutil
from functools import partial
from IPython.display import display

import pandas as pd
import torch
import wandb
from datasets import concatenate_datasets
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer, \
    AdamW, TrainingArguments

from pii_solution.cross_validation import split_new
from pii_solution.dataset_prepare import load_data, create_labels, create_dataset, renumber_dataset
from pii_solution.evaluation import PredictionNormalizationMode, make_prediction_one, calibrate
from pii_solution.metrics import compute_metrics
from pii_solution.model_tools import freeze_net, get_optimizer_params, UnfreezingCallback, init_clf_bias
from pii_solution.tokenizing import tokenize_train, tokenize_eval
from pii_solution.trainers.FocalLossTrainer import get_focal_loss_trainer_cls

os.environ["WANDB_PROJECT"] = "pii-detection-removal-from-educational-data"
os.environ["WANDB_WATCH"] = "all"

DIR = '/home/vladimir/competitions/pii-detection-removal-from-educational-data'
MODEL_BASE_DIR = f'{DIR}/models'
DATA_DIR = f'{DIR}/input'
F5_threshold = 0.925

MODEL_NAME = 'microsoft/deberta-v3-large'
MODEL_DIR = MODEL_BASE_DIR + '/' + MODEL_NAME.replace('/', '__').replace('-', '_')

# configs
TRAINING_MAX_LENGTH = 1750 #2048  # I use 1280 locally
TESTING_MAX_LENGTH = 3750  # I use 1280 locally
FREEZE_EMBEDDINGS = False
FREEZE_LAYERS = 0
USE_MIXTRAL_DATASET = True
USE_PJMATH_DATASET = False
USE_OPENPII_DATASET = False
CURRENT_FOLD = 5
NUM_TRAIN_EPOCHS = 8
NUM_EVALS_PER_EPOCH = 2
NUM_EXAMPLES_BATCH_FIT = 1
NUM_EXAMPLES_GRADIENT = 4
NUM_FOLDS = 10

FILTER_EXAMPLES_WITHOUT_POSITIVES = True
TRUNCATE_START_TO = 1750
DOUBLE_BASE_TRAIN_SET_TRUNCATE = True

USE_FOCAL_LOSS = True
focal_loss_gamma = 3
focal_loss_alpha_O = 0.02
WEIGHTED_LEARNING_RATE_DECAY = True
UNFREEZING_CALLBACK = False

data = load_data(f"{DATA_DIR}/train.json")
offset = max([row['document'] for row in data]) + 1

df = pd.DataFrame(data)[['document', 'tokens', 'labels']].copy()
df = df.explode(['tokens', 'labels']).reset_index(drop=True).rename(columns={'tokens': 'token_value', 'labels': 'label'})
df['token'] = df.groupby('document').cumcount()

data_mixtral = load_data(f"additional_data/moth/mixtral-8x7b-v1.json", offset) # https://www.kaggle.com/datasets/pjmathematician/pii-detection-dataset-gpt
offset = max([row['document'] for row in data_mixtral]) + 1

data_pjmath = load_data(f"additional_data/pjmath/pii_dataset_fixed.json", offset)  # https://www.kaggle.com/code/valentinwerner/fix-punctuation-tokenization-external-dataset/output
offset = max([row['document'] for row in data_pjmath]) + 1

data_openpii = load_data('additional_data/openpii_30k.jsonl', offset)
data_openpii = data_openpii[:len(data_openpii)//10]
offset = max([row['document'] for row in data_openpii]) + 1


all_labels, label2id, id2label = create_labels(data)

ds, ds_mixtral, ds_pjmath, ds_openpii = create_dataset(data), create_dataset(data_mixtral), create_dataset(data_pjmath), create_dataset(data_openpii)
ds_split_train_test = split_new(ds, fold_id=9, num_folds=10, random_state=77)
ds_train, ds_test = ds_split_train_test['train'], ds_split_train_test['test']
ds_split_train_val = split_new(ds_train, CURRENT_FOLD, NUM_FOLDS)
ds_train, ds_eval = ds_split_train_val['train'], ds_split_train_val['test']

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if DOUBLE_BASE_TRAIN_SET_TRUNCATE and TRUNCATE_START_TO is None:
    raise

ds_train_not_trunc = ds_train.map(
    tokenize_train,
    fn_kwargs={"tokenizer": tokenizer, "label2id": label2id, "max_length": TRAINING_MAX_LENGTH},
    num_proc=2
)
if TRUNCATE_START_TO is not None:
    ds_train_trunc_start = ds_train.map(renumber_dataset, fn_kwargs={'first_index': offset}, num_proc=2)
    ds_train_trunc_start = ds_train_trunc_start.map(
        tokenize_train,
        fn_kwargs={"tokenizer": tokenizer, "label2id": label2id, "max_length": TRAINING_MAX_LENGTH, 'truncate_left_input_to': TRUNCATE_START_TO},
        num_proc=2
    )
    if DOUBLE_BASE_TRAIN_SET_TRUNCATE:
        ds_train = concatenate_datasets([ds_train_not_trunc, ds_train_trunc_start]).shuffle(42)
    else:
        ds_train = ds_train_trunc_start
else:
    ds_train = ds_train_not_trunc

ds_train = ds_train.class_encode_column("group")

ds_eval = ds_eval.map(
    tokenize_train,
    fn_kwargs={"tokenizer": tokenizer, "label2id": label2id, "max_length": TRAINING_MAX_LENGTH, 'truncate_left_input_to': TRUNCATE_START_TO},
    num_proc=2
)
ds_eval = ds_eval.cast(ds_train.features)


ds_mixtral = ds_mixtral.map(
    tokenize_train,
    fn_kwargs={"tokenizer": tokenizer, "label2id": label2id, "max_length": TRAINING_MAX_LENGTH, 'truncate_left_input_to': TRUNCATE_START_TO},
    num_proc=2
)
ds_mixtral = ds_mixtral.cast(ds_train.features)

ds_pjmath = ds_pjmath.map(
    tokenize_train,
    fn_kwargs={"tokenizer": tokenizer, "label2id": label2id, "max_length": TRAINING_MAX_LENGTH, 'truncate_left_input_to': TRUNCATE_START_TO},
    num_proc=2
)
ds_pjmath = ds_pjmath.cast(ds_train.features)

ds_openpii = ds_openpii.map(
    tokenize_train,
    fn_kwargs={"tokenizer": tokenizer, "label2id": label2id, "max_length": TRAINING_MAX_LENGTH, 'truncate_left_input_to': TRUNCATE_START_TO},
    num_proc=2
)
ds_openpii = ds_openpii.cast(ds_train.features)

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(all_labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

#init_clf_bias(model, -2.0, 1.0)

num_layers = len(model.base_model.encoder.layer)
print('Num layers:', num_layers)
collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)
freeze_net(model, FREEZE_LAYERS, FREEZE_EMBEDDINGS)

if FILTER_EXAMPLES_WITHOUT_POSITIVES:
    ds_train = ds_train.filter(lambda x: x['group'] == 1)

if USE_MIXTRAL_DATASET:
    ds_train = concatenate_datasets([ds_train, ds_mixtral]).shuffle(42)

if USE_PJMATH_DATASET:
    ds_train = concatenate_datasets([ds_train, ds_pjmath]).shuffle(42)

if USE_OPENPII_DATASET:
    ds_train = concatenate_datasets([ds_train, ds_openpii]).shuffle(42)


trainer_cls = Trainer
if USE_FOCAL_LOSS:
    alpha = torch.ones(13)
    alpha[12] = focal_loss_alpha_O
    alpha = alpha.to('cuda:0')
    trainer_cls = get_focal_loss_trainer_cls(alpha, focal_loss_gamma)

if WEIGHTED_LEARNING_RATE_DECAY:
    params = get_optimizer_params(model, 5e-3, 1e-5, 1e-8, wd_rate=2)
    optimizer = AdamW(params)
    optimizer_args = (optimizer, None)
else:
    optimizer_args = (None, None)

callbacks = []
if UNFREEZING_CALLBACK:
    callbacks.append(UnfreezingCallback(min_layers=4, max_layers=int(num_layers * 0.75), num_epochs_unfreeze_embeddings=0))


additional_config = {
    'focal_loss_gamma': focal_loss_gamma,
    'focal_loss_alpha_O': focal_loss_alpha_O,
    'use_mixtral_dataset': USE_MIXTRAL_DATASET,
    'use_pjmath_dataset': USE_PJMATH_DATASET,
    'focal_loss': USE_FOCAL_LOSS,
    'freeze_embeddings': FREEZE_EMBEDDINGS,
    'freeze_layers': FREEZE_LAYERS,
    'filter_examples_without_positives': FILTER_EXAMPLES_WITHOUT_POSITIVES,
    'unfreezing_callback': UNFREEZING_CALLBACK,
    'fold': CURRENT_FOLD,
    'wlrd': WEIGHTED_LEARNING_RATE_DECAY,
    'num_folds': NUM_FOLDS,
    'double_base_set_truncate': DOUBLE_BASE_TRAIN_SET_TRUNCATE
}

wandb.init(config=additional_config)
eval_steps = 1/NUM_TRAIN_EPOCHS/NUM_EVALS_PER_EPOCH

run_path = f'{MODEL_DIR}/{wandb.run.name}'
args = TrainingArguments(
    output_dir=run_path,
    fp16=True,
    learning_rate=1e-4,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=NUM_EXAMPLES_BATCH_FIT,
    per_device_eval_batch_size=NUM_EXAMPLES_BATCH_FIT * 2,
    gradient_accumulation_steps=NUM_EXAMPLES_GRADIENT // NUM_EXAMPLES_BATCH_FIT,
    report_to="wandb",
    evaluation_strategy="steps",
    eval_steps=eval_steps,
    logging_strategy='steps',
    logging_steps=eval_steps,
    save_strategy='steps',
    save_steps=eval_steps,
    save_total_limit=1,
    overwrite_output_dir=True,
    load_best_model_at_end=True,
    lr_scheduler_type='cosine',
    metric_for_best_model="f1",
    greater_is_better=True,
    weight_decay=0.05,
    save_only_model=True,
    max_grad_norm=0.1
)

trainer = trainer_cls(
    model=model,
    args=args,
    train_dataset=ds_train,
    eval_dataset=ds_eval,
    data_collator=collator,
    tokenizer=tokenizer,
    compute_metrics=partial(compute_metrics, all_labels=all_labels),
    callbacks=callbacks,
    optimizers=optimizer_args
)

trainer.train()

ds_test = ds_test.map(tokenize_eval, fn_kwargs={"tokenizer": tokenizer, "max_length": TESTING_MAX_LENGTH}, num_proc=2)
real_data = df[df.document.isin(set(ds_test['document']))]

test_predictions = make_prediction_one(trainer.model, trainer.tokenizer, trainer.data_collator, ds_test)
preds = test_predictions.argmax(-1)
preds_without_O = test_predictions[:, :, :12].argmax(-1)
O_preds = test_predictions[:, :, 12]

calibration_results_nrw, tables_nrw = calibrate(ds_test, real_data, id2label, O_preds, preds_without_O, preds, 0.25, PredictionNormalizationMode.NO_REWRITE)
table_025 = tables_nrw['0.25']
display(table_025[(table_025.label_pred != table_025.label_real) & (table_025.label_real != 'O')])
calibration_results_rwi, tables_rwi = calibrate(ds_test, real_data, id2label, O_preds, preds_without_O, preds, 0.25, PredictionNormalizationMode.REWRITE_STARTING_I)
table_025 = tables_rwi['0.25']
display(table_025[(table_025.label_pred != table_025.label_real) & (table_025.label_real != 'O')])
calibration_results_rwb, tables_rwb = calibrate(ds_test, real_data, id2label, O_preds, preds_without_O, preds, 0.25, PredictionNormalizationMode.REWRITE_BOTH)
table_025 = tables_rwi['0.25']
display(table_025[(table_025.label_pred != table_025.label_real) & (table_025.label_real != 'O')])

wandb.log({f'calibration/{key}': value for key, value in calibration_results_rwi.items()})
wandb.unwatch()
wandb.finish()

best_f5 = calibration_results_rwi['best_f5']
BAD_RUN = best_f5 < F5_threshold

if BAD_RUN:
    print(f'Removing {run_path} cause of low val F5({best_f5:.5f} < {F5_threshold:.5f})')
    shutil.rmtree(run_path)
