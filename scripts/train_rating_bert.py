from __future__ import annotations

import argparse, inspect, json, shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, mean_absolute_error, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

SEED = 42
NUM_LABELS = 5
ID2LABEL = {i: f"{i+1} star" for i in range(5)}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "scripts" else Path.cwd()


def rpath(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def clean_text(s: pd.Series) -> pd.Series:
    return (s.fillna("").astype(str)
            .str.replace(r"https?://\S+|www\.\S+", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True).str.strip())


def text_key(s: pd.Series) -> pd.Series:
    x = clean_text(s).str.lower().str.replace(r"[^\w\s]", " ", regex=True)
    x = x.str.replace(r"\s+", " ", regex=True).str.strip()
    return pd.util.hash_pandas_object(x, index=False).astype("uint64").astype(str)


def round_half_up(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return np.floor(x + 0.5).clip(1, 5).astype("Int64")


def remove_conflicts(df: pd.DataFrame):
    before = len(df)
    n = df.groupby("rating_text_key", sort=False)["rating_class"].nunique()
    bad = set(n.index[n > 1])
    conflict_rows = int(df["rating_text_key"].isin(bad).sum())
    if bad:
        df = df[~df["rating_text_key"].isin(bad)].copy()
    b = len(df)
    df = df.drop_duplicates("rating_text_key", keep="first").copy()
    return df.reset_index(drop=True), {
        "rows_before": before,
        "conflicting_text_groups_removed": len(bad),
        "conflicting_rows_removed": conflict_rows,
        "duplicate_rows_removed": b - len(df),
        "rows_after": len(df),
    }


class RatingDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.texts = df["review_text"].tolist()
        self.labels = df["label_id"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tokenizer(self.texts[i], truncation=True, max_length=self.max_length, padding=False)
        enc["labels"] = self.labels[i]
        return enc


def metric_dict(y_true, y_pred):
    true_stars, pred_stars = y_true + 1, y_pred + 1
    err = np.abs(true_stars - pred_stars)
    return {
        "exact_accuracy": float(accuracy_score(y_true, y_pred)),
        "within_one_accuracy": float(np.mean(err <= 1)),
        "mae": float(mean_absolute_error(true_stars, pred_stars)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return metric_dict(labels.astype(int), np.argmax(logits, axis=1).astype(int))


def main():
    ap = argparse.ArgumentParser(description="Fine-tune the existing NLP Town five-star BERT on the canonical cleaned 20K dataset.")
    ap.add_argument("--input", default="data/processed/combined_multidomain_reviews.csv")
    ap.add_argument("--model-path", default="outputs/models/nlptown_bert_rating")
    ap.add_argument("--reports-dir", default="outputs/reports")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--learning-rate", type=float, default=1.5e-5)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--validation-size", type=float, default=0.15)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.10)
    args = ap.parse_args()

    set_seed(SEED)
    inp, model_path, reports = rpath(args.input), rpath(args.model_path), rpath(args.reports_dir)
    if not inp.exists(): raise FileNotFoundError(f"Missing canonical dataset: {inp}")

    df = pd.read_csv(inp, low_memory=False)
    need = {"domain", "review_text", "rating"}
    missing = need - set(df.columns)
    if missing: raise ValueError(f"Dataset missing columns: {sorted(missing)}")

    df["domain"] = df["domain"].fillna("").astype(str).str.strip().str.lower()
    df["review_text"] = clean_text(df["review_text"])
    df["rating_numeric"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df[df["review_text"].ne("") & df["rating_numeric"].between(1, 5)].copy()

    # NLP Town predicts integer 1..5 stars. Fractional hotel ratings are only
    # discretised for this 5-class target; canonical numeric ratings stay unchanged.
    df["rating_class"] = round_half_up(df["rating_numeric"]).astype(int)
    df["label_id"] = df["rating_class"] - 1
    df["rating_text_key"] = text_key(df["review_text"])
    df, cleanup = remove_conflicts(df)

    strata = df["domain"] + "__" + df["rating_class"].astype(str)
    if (strata.value_counts() < 2).any():
        raise ValueError("A domain/rating stratum is too small for stratified validation.")

    train_df, val_df = train_test_split(df, test_size=args.validation_size, random_state=SEED, stratify=strata)
    train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    overlap = set(train_df["rating_text_key"]) & set(val_df["rating_text_key"])
    if overlap: raise ValueError(f"Train/validation leakage: {len(overlap)} texts")

    print("\nNLP TOWN RATING-BERT — FRESH ORIGINAL CHECKPOINT")
    print("=" * 90)
    print(f"Canonical dataset: {inp}")
    print(f"Rows after cleanup: {len(df):,}")
    print(f"Train: {len(train_df):,} | Validation: {len(val_df):,}")
    print("Model: fresh nlptown/bert-base-multilingual-uncased-sentiment")
    print("Input column: review_text")
    print("Best checkpoint metric: exact_accuracy")
    print("Leakage check: PASS")
    print("\nRating distribution:")
    print(df["rating_class"].value_counts().sort_index().to_string())
    print("\nDomain x rating:")
    print(df.groupby(["domain", "rating_class"]).size().to_string())

    base_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=5)
    train_ds = RatingDataset(train_df, tokenizer, args.max_length)
    val_ds = RatingDataset(val_df, tokenizer, args.max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    reports.mkdir(parents=True, exist_ok=True)
    ckpt = reports / "_rating_bert_checkpoints"
    temp = model_path.parent / "_nlptown_bert_rating"
    shutil.rmtree(ckpt, ignore_errors=True); shutil.rmtree(temp, ignore_errors=True)

    kw = dict(output_dir=str(ckpt), num_train_epochs=args.epochs,
              per_device_train_batch_size=args.batch_size,
              per_device_eval_batch_size=max(16, args.batch_size),
              learning_rate=args.learning_rate, weight_decay=args.weight_decay,
              warmup_ratio=args.warmup_ratio, lr_scheduler_type="linear",
              optim="adamw_torch", save_strategy="epoch", logging_strategy="steps",
              logging_steps=100, load_best_model_at_end=True,
              metric_for_best_model="exact_accuracy", greater_is_better=True,
              save_total_limit=2, seed=SEED, data_seed=SEED, report_to=[],
              max_grad_norm=1.0, dataloader_num_workers=0,
              label_smoothing_factor=0.05, fp16=True, auto_find_batch_size=True)
    sig = inspect.signature(TrainingArguments.__init__)
    kw["eval_strategy" if "eval_strategy" in sig.parameters else "evaluation_strategy"] = "epoch"
    targs = TrainingArguments(**kw)

    tkw = dict(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
               data_collator=collator, compute_metrics=compute_metrics,
               callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.001)])
    tsig = inspect.signature(Trainer.__init__)
    tkw["processing_class" if "processing_class" in tsig.parameters else "tokenizer"] = tokenizer
    trainer = Trainer(**tkw)
    trainer.train()

    eval_metrics = trainer.evaluate(val_ds)
    pred = trainer.predict(val_ds)
    y_true = pred.label_ids.astype(int)
    y_pred = np.argmax(pred.predictions, axis=1).astype(int)
    fm = metric_dict(y_true, y_pred)

    trainer.save_model(str(temp)); tokenizer.save_pretrained(str(temp))

    metrics_path = reports / "phase7_rating_bert_metrics.json"
    matrix_path = reports / "phase7_rating_bert_confusion_matrix.csv"
    report_path = reports / "phase7_rating_bert_classification_report.txt"
    domain_path = reports / "phase7_rating_bert_per_domain_metrics.csv"
    history_path = reports / "phase7_rating_bert_training_history.csv"
    pred_path = reports / "phase7_rating_bert_validation_predictions.csv"

    payload = {
        "exact_rating_accuracy": fm["exact_accuracy"],
        "within_one_rating_accuracy": fm["within_one_accuracy"],
        "mean_absolute_error": fm["mae"],
        "macro_f1": fm["f1_macro"],
        "weighted_f1": fm["f1_weighted"],
        "precision_macro": fm["precision_macro"],
        "recall_macro": fm["recall_macro"],
        "eval_loss": float(eval_metrics.get("eval_loss", np.nan)),
        "training_input": str(inp), "train_rows": len(train_df), "validation_rows": len(val_df),
        "best_checkpoint": str(trainer.state.best_model_checkpoint or ""),
        "best_validation_exact_accuracy": float(trainer.state.best_metric) if trainer.state.best_metric is not None else None,
        "rating_target_rule": "nearest integer star using floor(rating + 0.5); canonical numeric rating remains unchanged",
        "cleanup": cleanup, "trained_at": datetime.now().isoformat(timespec="seconds"), "seed": SEED,
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    pd.DataFrame(confusion_matrix(y_true, y_pred, labels=[0,1,2,3,4]),
                 index=["actual_1","actual_2","actual_3","actual_4","actual_5"],
                 columns=["pred_1","pred_2","pred_3","pred_4","pred_5"]).to_csv(matrix_path)
    report_path.write_text(classification_report(y_true, y_pred, labels=[0,1,2,3,4],
        target_names=["1 star","2 stars","3 stars","4 stars","5 stars"], digits=4, zero_division=0), encoding="utf-8")

    rows=[]
    temp_eval = val_df[["domain"]].copy(); temp_eval["y_true"]=y_true; temp_eval["y_pred"]=y_pred
    for d,g in temp_eval.groupby("domain", sort=True): rows.append({"domain": d, "rows": len(g), **metric_dict(g.y_true.to_numpy(), g.y_pred.to_numpy())})
    pd.DataFrame(rows).to_csv(domain_path, index=False)
    pd.DataFrame([x for x in trainer.state.log_history if "eval_loss" in x]).to_csv(history_path, index=False)

    out = val_df[["domain","review_id","review_text","rating","rating_class"]].copy()
    out["predicted_star_rating"] = y_pred + 1
    out["absolute_error"] = np.abs(out["rating_class"] - out["predicted_star_rating"])
    out.to_csv(pred_path, index=False)

    # Deploy only after everything succeeded.
    if model_path.exists():
        shutil.rmtree(model_path)
    shutil.move(str(temp), str(model_path))
    shutil.rmtree(ckpt, ignore_errors=True)

    print("\nRATING-BERT COMPLETE — BEST CHECKPOINT SAVED")
    print("=" * 90)
    print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
    print(f"Exact accuracy:  {fm['exact_accuracy']*100:.2f}%")
    print(f"Within ±1 star:  {fm['within_one_accuracy']*100:.2f}%")
    print(f"MAE:             {fm['mae']:.4f}")
    print(f"Macro F1:        {fm['f1_macro']*100:.2f}%")
    print(f"Weighted F1:     {fm['f1_weighted']*100:.2f}%")
    print(f"Validation loss: {float(eval_metrics.get('eval_loss', np.nan)):.4f}")
    print(f"Final model:     {model_path}")
    print(f"Metrics:         {metrics_path}")

if __name__ == "__main__":
    main()