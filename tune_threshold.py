import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score

RUN_DIR = os.getcwd()
READY = os.path.join(RUN_DIR, "ready4TrainingData")

DATA_PARQ = os.path.join(READY, "CrashRisk_Training_2024_2025.parquet")
MODEL = os.path.join(READY, "xgb_crash_classifier_strict_time.json")

TIME_COL = "ROLLUP_TIMESTAMP"
LABEL_COL = "label"
NY_TZ = "America/New_York"

LEAK_COLS = [
    "EVENT_ID", "detected_dt", "distance_meters",
    "ARCHIVE_DETECTOR_ID", "ARCHIVE_LINK_ID", "ARCHIVE_LANE_ID",
    "DETECTOR_NAME", "DIRECTION", "EQUIP_LOC_ROADWAY",
]
ID_COLS = ["LANE_NAME", "SID", "SEGMENTS", "CONDITIONS"]

def main():
    df = pd.read_parquet(DATA_PARQ)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df[df[TIME_COL].notna()].copy()
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    # same strict split as training: last 90 days test
    max_ts = df[TIME_COL].max()
    test_start = max_ts - pd.Timedelta(days=90)
    test_df = df[df[TIME_COL] >= test_start].copy()

    drop_cols = set([TIME_COL, LABEL_COL]) | set(ID_COLS) | set(LEAK_COLS)
    drop_cols = [c for c in drop_cols if c in df.columns]

    y = test_df[LABEL_COL].values
    X = test_df.drop(columns=drop_cols, errors="ignore")
    X = X[[c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]].replace([np.inf,-np.inf], np.nan).fillna(0.0)

    booster = xgb.Booster()
    booster.load_model(MODEL)

    dtest = xgb.DMatrix(X, feature_names=list(X.columns))
    p = booster.predict(dtest)

    print("AUC:", round(roc_auc_score(y, p), 4))
    print("PR-AUC:", round(average_precision_score(y, p), 4))

    # sweep thresholds
    thrs = np.linspace(0.05, 0.95, 19)
    rows = []
    for t in thrs:
        yhat = (p >= t).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
        alert_rate = yhat.mean()
        rows.append((t, prec, rec, f1, alert_rate))

    out = pd.DataFrame(rows, columns=["thr", "precision", "recall", "f1", "alert_rate"])
    out["alerts_per_day"] = out["alert_rate"] * len(y) / 90.0

    print("\nThreshold sweep:")
    print(out.to_string(index=False))

    best_f1 = out.iloc[out["f1"].idxmax()]
    print("\nBest F1 threshold:", best_f1.to_dict())

    # Precision@K style: top-N alerts/day
    # Example: allow ~10 alerts/day
    allowed_per_day = 10
    allowed_total = int(allowed_per_day * 90)
    idx = np.argsort(-p)  # descending
    top = np.zeros_like(y)
    top[idx[:allowed_total]] = 1
    prec, rec, f1, _ = precision_recall_fscore_support(y, top, average="binary", zero_division=0)
    print(f"\nTop-{allowed_per_day} alerts/day policy (~{allowed_total} alerts in 90d):")
    print(" precision:", round(prec,4), "recall:", round(rec,4), "f1:", round(f1,4))

if __name__ == "__main__":
    main()
