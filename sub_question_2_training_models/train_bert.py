"""
BERT Baseline - Multi-label Frame Classification
Fine-tunes bert-base-uncased on the golden standard dataset.
Input columns: comment_body (text), gold_frames (comma-separated labels)
Split: 80% train / 20% test
"""

import ast
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
import torch
from torch import nn

# Configuration
DATA_PATH = "golden_standard.csv"
MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "./bert_checkpoints"
MAX_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
WEIGHT_DECAY = 0.01
THRESHOLD = 0.5
RANDOM_STATE = 42

# Reproducibility
from transformers import set_seed
set_seed(RANDOM_STATE)

# Load and preprocess data
def parse_gold_frames(value: str) -> list[str]:
    """Parse gold_frames column to a Python list of label strings."""
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
        return [item.strip() for item in value.split(",")]
    return value


print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Total rows: {len(df)}")

# Parse labels and drop rows with missing text or labels
df["gold_frames"] = df["gold_frames"].apply(parse_gold_frames)
df = df.dropna(subset=["comment_body", "gold_frames"])

# Encode labels to multi-hot binary vectors
label_encoder = MultiLabelBinarizer()
binary_labels = label_encoder.fit_transform(df["gold_frames"])
df["labels"] = list(binary_labels.astype(float))

num_labels = len(label_encoder.classes_)
print(f"Number of unique labels: {num_labels}")
print(f"Labels: {label_encoder.classes_}")

# Train / validation / test split.
# Test is the SAME 20% held out as before (seed 42); validation is carved from the rest
trainval_df, test_df = train_test_split(df, test_size=0.20, random_state=RANDOM_STATE)
train_df, val_df = train_test_split(trainval_df, test_size=0.1875, random_state=RANDOM_STATE)
print(f"Training   samples : {len(train_df)}")
print(f"Validation samples : {len(val_df)}")
print(f"Testing    samples : {len(test_df)}")

# Class-balanced positive weights: prevents the model collapsing to all-zero predictions.
_label_mat = np.array(train_df["labels"].tolist(), dtype=np.float32)
_pos = _label_mat.sum(axis=0)
_neg = _label_mat.shape[0] - _pos
POS_WEIGHT = torch.tensor(np.clip(_neg / np.maximum(_pos, 1.0), 1.0, 50.0), dtype=torch.float)

# Convert to HuggingFace Datasets (train / validation / test)
def _to_ds(d):
    return Dataset.from_pandas(
        d[["comment_body", "labels"]].rename(columns={"comment_body": "text"}),
        preserve_index=False,
    )
dataset = DatasetDict({"train": _to_ds(train_df), "validation": _to_ds(val_df), "test": _to_ds(test_df)})

# Tokenization
print(f"\nLoading tokenizer for '{MODEL_NAME}'...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )


tokenized_dataset = dataset.map(tokenize, batched=True)

# Model initialization
print(f"\nInitializing model '{MODEL_NAME}' for multi-label classification...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    problem_type="multi_label_classification",
)

# Metrics
def compute_metrics(pred):
    logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    probs = 1 / (1 + np.exp(-logits))          
    preds = (probs >= THRESHOLD).astype(int)
    labels = pred.label_ids.astype(int)

    metrics = {
        "accuracy":        accuracy_score(labels, preds),
        "f1_micro":        f1_score(labels, preds, average="micro",  zero_division=0),
        "precision_micro": precision_score(labels, preds, average="micro", zero_division=0),
        "recall_micro":    recall_score(labels, preds, average="micro",    zero_division=0),
        "f1_macro":        f1_score(labels, preds, average="macro",  zero_division=0),
    }

    # Per-class metrics
    f1_per_class        = f1_score(labels, preds, average=None, zero_division=0)
    precision_per_class = precision_score(labels, preds, average=None, zero_division=0)
    recall_per_class    = recall_score(labels, preds, average=None, zero_division=0)
    for i, label in enumerate(label_encoder.classes_):
        metrics[f"f1_{label}"]        = f1_per_class[i]
        metrics[f"precision_{label}"] = precision_per_class[i]
        metrics[f"recall_{label}"]    = recall_per_class[i]

    return metrics

# Training
class WeightedTrainer(Trainer):
    """Trainer with class-balanced BCE loss to avoid all-zero multi-label predictions."""
    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        pw = self._pos_weight.to(logits.device) if self._pos_weight is not None else None
        loss = nn.BCEWithLogitsLoss(pos_weight=pw)(logits, labels.float())
        return (loss, outputs) if return_outputs else loss


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_micro",
    fp16=True,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=50,
    save_total_limit=1,
    save_only_model=True,
    seed=RANDOM_STATE,
    data_seed=RANDOM_STATE,
    report_to="none",
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics,
    pos_weight=POS_WEIGHT,
)

print("\nStarting training...")
trainer.train()

# Save model + validation-tuned evaluation on the test set
SAVE_PATH = OUTPUT_DIR.replace("_checkpoints", "_final_model")
trainer.save_model(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print(f"\nModel saved to '{SAVE_PATH}'")

import eval_utils
_prefix = OUTPUT_DIR.replace("./", "").replace("_checkpoints", "")
eval_utils.report(trainer, tokenized_dataset, list(label_encoder.classes_), _prefix)
