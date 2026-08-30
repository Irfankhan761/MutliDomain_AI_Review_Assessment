"""
Agent 3: Transformer Sentiment Agent
Model: Fine-tuned DistilBERT-base-uncased

This upgraded version is used in Phase 6 final training.
It supports:
- DistilBERT fine-tuning
- weighted loss for class imbalance
- CPU/GPU automatic selection
- prediction labels + confidence scores
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

BASE_MODEL_NAME = "distilbert-base-uncased"

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class ReviewDataset(Dataset):
    """Torch dataset wrapping tokenized review texts and labels."""

    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
    }


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross entropy to reduce majority-class dominance."""

    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)

        loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class SentimentAnalysisAgent:
    """
    Fine-tunes / loads a DistilBERT sentiment classifier and predicts
    negative / neutral / positive labels for review text.
    """

    def __init__(
        self,
        model_path: str = "outputs/models/distilbert_sentiment",
        base_model_name: str = BASE_MODEL_NAME,
        max_length: int = 128,
    ):
        self.model_path = Path(model_path)
        self.base_model_name = base_model_name
        self.max_length = max_length
        self.labels = ["negative", "neutral", "positive"]
        self.tokenizer = None
        self.model = None

    def _tokenize(self, texts):
        return self.tokenizer(
            list(texts),
            truncation=True,
            padding=False,
            max_length=self.max_length,
        )

    def _calculate_class_weights(self, y_train):
        counts = np.bincount(y_train, minlength=3).astype(float)
        total = counts.sum()
        weights = []
        for count in counts:
            if count == 0:
                weights.append(0.0)
            else:
                weights.append(total / (len(counts) * count))
        return torch.tensor(weights, dtype=torch.float)

    def train(
        self,
        df: pd.DataFrame,
        text_column: str = "clean_review",
        label_column: str = "sentiment_label",
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        sample_size: int = None,
        output_dir: str = "outputs/models/distilbert_sentiment_checkpoints",
    ):
        """
        Fine-tunes DistilBERT-base-uncased on labelled review data.
        sample_size=None means use all supplied rows.
        """
        df = df.copy().dropna(subset=[text_column, label_column])
        df[text_column] = df[text_column].astype(str).str.strip()
        df = df[df[text_column] != ""].copy()
        df = df[df[label_column].isin(LABEL2ID.keys())].copy()

        if sample_size is not None and sample_size < len(df):
            per_class = max(1, sample_size // df[label_column].nunique())
            parts = [
                group.sample(min(len(group), per_class), random_state=42)
                for _, group in df.groupby(label_column)
            ]
            df = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)

        X_train, X_val, y_train_labels, y_val_labels = train_test_split(
            df[text_column].tolist(),
            df[label_column].tolist(),
            test_size=0.15,
            random_state=42,
            stratify=df[label_column],
        )

        y_train = [LABEL2ID[label] for label in y_train_labels]
        y_val = [LABEL2ID[label] for label in y_val_labels]
        class_weights = self._calculate_class_weights(y_train)

        print("\nDistilBERT training label distribution:")
        print(pd.Series(y_train_labels).value_counts())
        print("Class weights [negative, neutral, positive]:", class_weights.tolist())
        print("Device:", "cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.base_model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.base_model_name,
            num_labels=3,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )

        train_encodings = self._tokenize(X_train)
        val_encodings = self._tokenize(X_val)

        train_dataset = ReviewDataset(train_encodings, y_train)
        val_dataset = ReviewDataset(val_encodings, y_val)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # training_args = TrainingArguments(
        #     output_dir=output_dir,
        #     num_train_epochs=num_epochs,
        #     per_device_train_batch_size=batch_size,
        #     per_device_eval_batch_size=batch_size,
        #     learning_rate=learning_rate,
        #     eval_strategy="epoch",
        #     save_strategy="no",
        #     logging_steps=50,
        #     report_to=[],
        #     use_cpu=not torch.cuda.is_available(),
        # )
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,

            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,

            learning_rate=learning_rate,

            # Optimisation
            optim="adamw_torch",
            weight_decay=0.01,
            warmup_ratio=0.10,
            lr_scheduler_type="linear",

            # Evaluate and save every epoch
            eval_strategy="epoch",
            save_strategy="epoch",

            # Keep the epoch with the best Macro F1
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            save_total_limit=2,

            logging_steps=50,
            report_to=[],

            use_cpu=not torch.cuda.is_available(),
        )


        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        eval_metrics = trainer.evaluate()

        preds_output = trainer.predict(val_dataset)
        y_pred = np.argmax(preds_output.predictions, axis=-1)

        report = classification_report(
            y_val, y_pred, target_names=self.labels, zero_division=0
        )
        matrix = confusion_matrix(y_val, y_pred, labels=[0, 1, 2])

        self.model_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(self.model_path)
        self.tokenizer.save_pretrained(self.model_path)

        return eval_metrics, report, matrix

    def load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained DistilBERT sentiment model not found at {self.model_path}. Run train() first."
            )
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(self.model_path)
        self.model.eval()

    def predict(self, texts, batch_size: int = 32):
        if self.model is None or self.tokenizer is None:
            self.load()

        texts = [str(text) if text is not None else "" for text in list(texts)]
        all_preds = []
        all_confidences = []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                encodings = self.tokenizer(
                    batch,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(device)

                logits = self.model(**encodings).logits
                probs = torch.softmax(logits, dim=-1)
                confidences, preds = torch.max(probs, dim=-1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_confidences.extend(confidences.cpu().numpy().tolist())

        labels = [ID2LABEL[p] for p in all_preds]
        return labels, all_confidences
