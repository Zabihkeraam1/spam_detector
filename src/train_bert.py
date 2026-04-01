import pandas as pd
import torch
import numpy as np

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import CrossEntropyLoss

DATA_PATH = "data/processed/spam_cleaned.csv"
data = pd.read_csv(DATA_PATH)

data = data.dropna(subset=['message', 'label'])
data['message'] = data['message'].astype(str)
data['label'] = data['label'].astype(int)

print("\nLabel distribution:\n", data['label'].value_counts())

train_df, test_df = train_test_split(
    data,
    test_size=0.2,
    random_state=42,
    stratify=data['label']
)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(example):
    return tokenizer(
        example["message"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.remove_columns(["message"])
test_dataset = test_dataset.remove_columns(["message"])

train_dataset.set_format("torch")
test_dataset.set_format("torch")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

model.config.id2label = {0: "HAM", 1: "SPAM"}
model.config.label2id = {"HAM": 0, "SPAM": 1}

y = data['label'].values

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y),
    y=y
)

weights = torch.tensor(class_weights, dtype=torch.float)
print("\nClass Weights:", weights)

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = CrossEntropyLoss(weight=weights.to(logits.device))
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)

    return {"accuracy": acc, "f1": f1}

training_args = TrainingArguments(
    output_dir="./bert_results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

results = trainer.evaluate()
print("\nFinal Results:", results)

trainer.save_model("models/bert_model")
tokenizer.save_pretrained("models/bert_model")

print("\n✅ BERT model saved correctly!")