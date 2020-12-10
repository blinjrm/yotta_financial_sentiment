"""Utilities to train the model.

"""

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split


class SentimentDataset(torch.utils.data.Dataset):
    """Dataset object used by the pytorch model"""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def create_dataset(X, y, tokenizer):
    """Creates a dataset to be passed to the model

    Args:
        X (list of strings): list of headlines
        y (list of ints): list of sentiment (0='neutral', 1='positive', 2='negative')
        tokenizer (huggingface tokenizer): the tokenizer to be used to tokenize the data

    Returns:
        Dataset: data ready to be used in the pytorch model
    """

    train_texts, test_texts, train_labels, test_labels = train_test_split(X, y, test_size=0.15)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    train_dataset = SentimentDataset(train_encodings, train_labels)
    test_dataset = SentimentDataset(test_encodings, test_labels)

    return train_dataset, test_dataset


def compute_metrics(pred):
    """Compute the metrics for the model, when calling model.evaluate()

    Args:
        pred (int): prediction for the sentiment of a headline

    Returns:
        dict: dictionnary containing the chosen metrics
    """

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }
