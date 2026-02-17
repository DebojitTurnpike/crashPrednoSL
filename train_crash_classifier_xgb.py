import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix

RUN_DIR = os.getcwd()
READY = os.path.join(RUN_DIR, "ready4TrainingData")
DATA_PARQ = os.path.join(READY, "CrashRisk_Training_2024_2025.parquet")
MODEL_OUT = os.path.join(READY, "xgb_crash_classifier_strict_time.json")

TIME_COL = "ROLLUP_TIMESTAMP"
LABEL_COL = "label"

# Crash-only / leakage-prone cols (drop if present)
LEAK_COLS = [
    "EVENT_ID", "detected_dt", "distance_meters",
    "ARCHIVE_DETECTOR_ID", "ARCHIVE_LINK_ID", "ARCHIVE_LANE_ID",
    "DETECTOR_NAME", "DIRECTION", "EQUIP_LOC_ROADWAY",
]

# Pure identifiers (drop)
ID_COLS = ["LANE_NAME", "SID", "SEGMENTS", "CONDITIONS"]  # CONDITIONS is text; keep out of XGB unless encoded

def main():
    if not os.path.exists(DATA_PARQ):
        raise FileNotFoundError(DATA_PARQ)

    df = pd.read_parquet(DATA_PARQ)

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df[df[TIME_COL].notna()].copy()
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    # STRICT time split: last 90 days as test
    max_ts = df[TIME_COL].max()
    test_start = max_ts - pd.Timedelta(days=90)

    train_df = df[df[TIME_COL] < test_start].copy()
    test_df  = df[df[TIME_COL] >= test_start].copy()

    print("Time split (STRICT):")
    print("  train:", train_df[TIME_COL].min(), "to", train_df[TIME_COL].max(), "| rows:", len(train_df))
    print("  test :", test_df[TIME_COL].min(),  "to", test_df[TIME_COL].max(),  "| rows:", len(test_df))

    # Drop leakage + ids + timestamp + label
    drop_cols = set([TIME_COL, LABEL_COL]) | set(ID_COLS) | set(LEAK_COLS)
    drop_cols = [c for c in drop_cols if c in df.columns]
    print("\nDropping columns:", drop_cols)

    def prep_xy(d):
        y = d[LABEL_COL].values
        X = d.drop(columns=drop_cols, errors="ignore")

        # numeric only
        num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        X = X[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return X, y

    X_train, y_train = prep_xy(train_df)
    X_test, y_test = prep_xy(test_df)

    # Align columns
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)

    # Imbalance
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    spw = neg / max(pos, 1)
    print("\nTrain label counts:", {0: neg, 1: pos})
    print("scale_pos_weight:", round(spw, 3))
    print("Features used:", X_train.shape[1])

    params = dict(
        objective="binary:logistic",
        eval_metric="auc",
        max_depth=6,
        eta=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        tree_method="hist",
        scale_pos_weight=spw,
    )

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(X_train.columns))
    dtest  = xgb.DMatrix(X_test,  label=y_test,  feature_names=list(X_train.columns))

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        evals=[(dtrain, "train"), (dtest, "test")],
        early_stopping_rounds=100,
        verbose_eval=100,
    )

    p = model.predict(dtest)

    if len(np.unique(y_test)) > 1:
        auc = roc_auc_score(y_test, p)
        ap  = average_precision_score(y_test, p)
    else:
        auc, ap = float("nan"), float("nan")

    print("\n=== TEST METRICS (STRICT last 90 days) ===")
    print("AUC:", round(auc, 4))
    print("PR-AUC:", round(ap, 4))

    thr = 0.5
    yhat = (p >= thr).astype(int)
    print("\nConfusion matrix @0.5:")
    print(confusion_matrix(y_test, yhat))
    print("\nReport:")
    print(classification_report(y_test, yhat, digits=4))

    model.save_model(MODEL_OUT)
    print("\n✅ Saved model:", MODEL_OUT)

if __name__ == "__main__":
    main()
