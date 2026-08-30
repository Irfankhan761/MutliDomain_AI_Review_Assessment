from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, mean_absolute_error, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "data" / "processed" / "combined_multidomain_reviews.csv"
MODEL_PATH = ROOT / "outputs" / "models" / "nlptown_bert_rating"
REPORT_DIR = ROOT / "outputs" / "reports"
MAX_LENGTH = 256
BATCH_SIZE = 32
VALIDATION_SIZE = 0.15


def clean_text(series):
    return (series.fillna("").astype(str)
            .str.replace(r"https?://\S+|www\.\S+", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True).str.strip())


def text_key(series):
    x = clean_text(series).str.lower().str.replace(r"[^\w\s]", " ", regex=True)
    x = x.str.replace(r"\s+", " ", regex=True).str.strip()
    return pd.util.hash_pandas_object(x, index=False).astype("uint64").astype(str)


def round_half_up(series):
    x = pd.to_numeric(series, errors="coerce")
    return np.floor(x + 0.5).clip(1, 5).astype("Int64")


def remove_conflicts_and_duplicates(df):
    before = len(df)
    label_counts = df.groupby("rating_text_key", sort=False)["rating_class"].nunique()
    conflict_keys = set(label_counts.index[label_counts > 1])
    conflict_rows = int(df["rating_text_key"].isin(conflict_keys).sum())
    if conflict_keys:
        df = df[~df["rating_text_key"].isin(conflict_keys)].copy()
    before_dedup = len(df)
    df = df.drop_duplicates("rating_text_key", keep="first").copy()
    duplicate_rows_removed = before_dedup - len(df)
    return df.reset_index(drop=True), {
        "rows_before_cleanup": int(before),
        "conflicting_text_groups_removed": int(len(conflict_keys)),
        "conflicting_rows_removed": int(conflict_rows),
        "duplicate_rows_removed": int(duplicate_rows_removed),
        "rows_after_cleanup": int(len(df)),
    }


def predict(model, tokenizer, texts, device):
    preds, confidences = [], []
    model.to(device); model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start:start+BATCH_SIZE]
            inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH)
            inputs = {k:v.to(device) for k,v in inputs.items()}
            probs = torch.softmax(model(**inputs).logits, dim=1)
            pred = torch.argmax(probs, dim=1)
            preds.extend((pred.cpu().numpy()+1).astype(int).tolist())
            confidences.extend(probs.max(dim=1).values.cpu().numpy().astype(float).tolist())
    return np.asarray(preds, dtype=int), np.asarray(confidences, dtype=float)


def metric_dict(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int); y_pred = np.asarray(y_pred, dtype=int)
    errors = np.abs(y_true-y_pred)
    return {
        "exact_accuracy": float(accuracy_score(y_true, y_pred)),
        "within_one_star_accuracy": float(np.mean(errors <= 1)),
        "mean_absolute_error": float(mean_absolute_error(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "severe_error_rate_gap_3plus": float(np.mean(errors >= 3)),
        "four_star_error_rate": float(np.mean(errors == 4)),
    }


def main():
    if not DATA_PATH.exists(): raise FileNotFoundError(DATA_PATH)
    if not MODEL_PATH.exists(): raise FileNotFoundError(MODEL_PATH)

    print("\nFINAL NLP TOWN RATING-BERT VALIDATION")
    print("="*92)
    print("This script DOES NOT retrain or modify the model.")
    print("Dataset:", DATA_PATH)
    print("Model:  ", MODEL_PATH)
    print(f"Split: 85/15 stratified by domain + rating, seed={SEED}")
    print(f"Max length: {MAX_LENGTH}")

    df = pd.read_csv(DATA_PATH, low_memory=False)
    required = {"domain","review_text","rating"}
    missing = required - set(df.columns)
    if missing: raise ValueError(f"Dataset missing columns: {sorted(missing)}")
    df["domain"] = df["domain"].fillna("").astype(str).str.strip().str.lower()
    df["review_text"] = clean_text(df["review_text"])
    df["rating_numeric"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df[df["review_text"].ne("") & df["rating_numeric"].between(1,5)].copy()
    df["rating_class"] = round_half_up(df["rating_numeric"]).astype(int)
    df["rating_text_key"] = text_key(df["review_text"])
    df, cleanup = remove_conflicts_and_duplicates(df)

    strata = df["domain"] + "__" + df["rating_class"].astype(str)
    train_df, val_df = train_test_split(df, test_size=VALIDATION_SIZE, random_state=SEED, stratify=strata)
    train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    overlap = set(train_df["rating_text_key"]) & set(val_df["rating_text_key"])
    if overlap: raise ValueError(f"Train/validation leakage detected: {len(overlap)}")

    print(f"\nRaw dataset rows:      {cleanup['rows_before_cleanup']:,}")
    print(f"Rows after rating QC:  {cleanup['rows_after_cleanup']:,}")
    print(f"Training split rows:   {len(train_df):,}")
    print(f"Validation split rows: {len(val_df):,}")
    print("Leakage check:         PASS")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH), local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    y_true = val_df["rating_class"].astype(int).to_numpy()
    y_pred, confidence = predict(model, tokenizer, val_df["review_text"].tolist(), device)
    metrics = metric_dict(y_true, y_pred)

    print("\nFINAL HELD-OUT RESULTS")
    print("="*92)
    print(f"Exact 5-star accuracy:       {metrics['exact_accuracy']*100:.2f}%")
    print(f"Within ±1 star accuracy:     {metrics['within_one_star_accuracy']*100:.2f}%")
    print(f"Mean Absolute Error (MAE):   {metrics['mean_absolute_error']:.4f} stars")
    print(f"Macro F1:                    {metrics['macro_f1']*100:.2f}%")
    print(f"Weighted F1:                 {metrics['weighted_f1']*100:.2f}%")
    print(f"Macro Precision:             {metrics['macro_precision']*100:.2f}%")
    print(f"Macro Recall:                {metrics['macro_recall']*100:.2f}%")
    print(f"Severe error rate (gap >=3): {metrics['severe_error_rate_gap_3plus']*100:.3f}%")
    print(f"4-star extreme error rate:   {metrics['four_star_error_rate']*100:.3f}%")

    conf = pd.DataFrame(confusion_matrix(y_true,y_pred,labels=[1,2,3,4,5]),
                        index=["actual_1","actual_2","actual_3","actual_4","actual_5"],
                        columns=["pred_1","pred_2","pred_3","pred_4","pred_5"])

    domain_rows=[]
    temp = val_df[["domain"]].copy(); temp["actual_star"]=y_true; temp["predicted_star"]=y_pred
    for domain,g in temp.groupby("domain", sort=True):
        domain_rows.append({"domain":domain,"rows":len(g),**metric_dict(g["actual_star"],g["predicted_star"])})
    domain_df = pd.DataFrame(domain_rows)

    pred_cols=[c for c in ["domain","review_id","entity_id","entity_name","review_text","rating","rating_class"] if c in val_df.columns]
    pred_df=val_df[pred_cols].copy(); pred_df["predicted_star_rating"]=y_pred; pred_df["prediction_confidence"]=confidence; pred_df["absolute_error"]=np.abs(y_true-y_pred)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path=REPORT_DIR/"final_rating_model_validation.json"
    confusion_path=REPORT_DIR/"final_rating_model_confusion_matrix.csv"
    domain_path=REPORT_DIR/"final_rating_model_domain_metrics.csv"
    predictions_path=REPORT_DIR/"final_rating_model_validation_predictions.csv"
    class_report_path=REPORT_DIR/"final_rating_model_classification_report.txt"

    payload={"evaluation_type":"held_out_validation_reproducing_training_split_logic","dataset":str(DATA_PATH),"model":str(MODEL_PATH),"seed":SEED,"validation_fraction":VALIDATION_SIZE,"max_length":MAX_LENGTH,"train_rows":int(len(train_df)),"validation_rows":int(len(val_df)),"cleanup":cleanup,**metrics}
    metrics_path.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    conf.to_csv(confusion_path); domain_df.to_csv(domain_path,index=False); pred_df.to_csv(predictions_path,index=False)
    class_report_path.write_text(classification_report(y_true,y_pred,labels=[1,2,3,4,5],target_names=["1 star","2 stars","3 stars","4 stars","5 stars"],digits=4,zero_division=0),encoding="utf-8")

    print("\nCONFUSION MATRIX")
    print(conf.to_string())
    print("\nPER-DOMAIN RESULTS")
    d=domain_df.copy()
    for c in ["exact_accuracy","within_one_star_accuracy","macro_f1"]: d[c]=(d[c]*100).round(2)
    print(d[["domain","rows","exact_accuracy","within_one_star_accuracy","mean_absolute_error","macro_f1"]].to_string(index=False))
    print("\nSaved final dissertation evidence:")
    for p in [metrics_path,confusion_path,domain_path,predictions_path,class_report_path]: print(" -",p)

if __name__ == "__main__":
    main()
