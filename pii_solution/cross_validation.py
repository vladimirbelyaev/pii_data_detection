from sklearn.model_selection import KFold
from datasets import DatasetDict, Dataset


def split_new(ds: Dataset, fold_id: int, num_folds: int = 5, random_state=42):
    documents = [doc['document'] for doc in ds]
    kf = KFold(n_splits=num_folds, random_state=random_state, shuffle=True)

    train_doc_ids, test_doc_ids = None, None
    for idx, (train_doc_ids, test_doc_ids) in enumerate(kf.split(documents)):
        if idx == fold_id:
            print(len(train_doc_ids), len(test_doc_ids))
            break

    if train_doc_ids is None or test_doc_ids is None:
        raise
    train_docs = set([documents[i] for i in train_doc_ids])
    test_docs = set([documents[i] for i in test_doc_ids])
    return DatasetDict(
        {
            'train': ds.filter(lambda x: x['document'] in train_docs),
            'test': ds.filter(lambda x: x['document'] in test_docs)
        }
    )


def split_old(ds: Dataset):
    return ds.train_test_split(test_size=0.2, seed=42)
