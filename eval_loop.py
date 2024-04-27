import pandas as pd
from IPython.display import display
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification


from pii_solution.cross_validation import split_new, split_old
from pii_solution.dataset_prepare import create_dataset, create_labels, load_data
from pii_solution.evaluation import PredictionNormalizationMode, calibrate, make_prediction_one
from pii_solution.tokenizing import tokenize_eval


DIR = '/home/vladimir/competitions/pii-detection-removal-from-educational-data'
MODEL_BASE_DIR = f'{DIR}/models'
DATA_DIR = f'{DIR}/input'

#MODEL_NAME = 'microsoft_deberta-v3-large/checkpoint-5400'
MODEL_NAME = f'{MODEL_BASE_DIR}/microsoft__deberta_v3_large/bumbling-lake-95/checkpoint-5400'

MODEL_DIR = MODEL_NAME
TESTING_MAX_LENGTH = 3750  # I use 1280 locally


data = load_data(f"{DATA_DIR}/train.json")
df = pd.DataFrame(data)[['document', 'tokens', 'labels']].copy()
df = df.explode(['tokens', 'labels']).reset_index(drop=True).rename(columns={'tokens': 'token_value', 'labels': 'label'})
df['token'] = df.groupby('document').cumcount()
ds = create_dataset(data)
ds_split = split_old(ds)
ds_split = split_new(ds, 2)

all_labels, label2id, id2label = create_labels(data)

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(all_labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)

ds_test = ds_split['test']
ds_test = ds_test.map(
    tokenize_eval,
    fn_kwargs={"tokenizer": tokenizer, "max_length": TESTING_MAX_LENGTH},
    num_proc=2
)
#ds_test = ds_test.class_encode_column("group")

predictions = make_prediction_one(model, tokenizer, collator, ds_test)


preds = predictions.argmax(-1)
preds_without_O = predictions[:,:,:12].argmax(-1)
O_preds = predictions[:,:,12]


real_data = df
real_data = real_data[real_data.document.isin(set(ds_test['document']))]

calibration_results, tables = calibrate(ds_test, real_data, id2label, O_preds, preds_without_O, preds, 0.25, PredictionNormalizationMode.NO_REWRITE)

some_table = tables['0.25']
display(some_table[(some_table.label_pred != some_table.label_real) & (some_table.label_real != 'O')])


calibration_results, tables = calibrate(ds_test, real_data, id2label, O_preds, preds_without_O, preds, 0.25, PredictionNormalizationMode.REWRITE_STARTING_I)

some_table = tables['0.25']

display(some_table[(some_table.label_pred != some_table.label_real) & (some_table.label_real != 'O')])


calibration_results, tables = calibrate(ds_test, real_data, id2label, O_preds, preds_without_O, preds, 0.25, PredictionNormalizationMode.REWRITE_BOTH)

some_table = tables['0.25']

display(some_table[(some_table.label_pred != some_table.label_real) & (some_table.label_real != 'O')])