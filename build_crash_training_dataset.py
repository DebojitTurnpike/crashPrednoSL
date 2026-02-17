import os
import pandas as pd
import numpy as np

NY_TZ = "America/New_York"

RUN_DIR = os.getcwd()
READY = os.path.join(RUN_DIR, "ready4TrainingData")

CRASH_PARQ = os.path.join(READY, "CrashData_WithWeather_2024_2025_FINAL.parquet")
NONCRASH_PARQ = os.path.join(READY, "NonCrash_Sample10pct_2024_2025_CLEAN.parquet")
PRED_PARQ = os.path.join(READY, "Timeseries_Predictions_2024_2025.parquet")

OUT_PARQ = os.path.join(READY, "CrashRisk_Training_2024_2025.parquet")
OUT_XLSX = os.path.join(READY, "CrashRisk_Training_2024_2025.xlsx")

TIME_COL = "ROLLUP_TIMESTAMP"
LANE_COL = "LANE_NAME"
SID_COL  = "SID"

def must_exist(p):
    if not os.path.exists(p):
        raise FileNotFoundError(p)

def floor15_ny(ts: pd.Series) -> pd.Series:
    """Ensure tz-aware NY and floor to 15-min safely (floor in UTC to avoid DST issues)."""
    ts = pd.to_datetime(ts, errors="coerce")
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(NY_TZ, ambiguous="infer", nonexistent="shift_forward")
    else:
        ts = ts.dt.tz_convert(NY_TZ)
    ts_utc = ts.dt.tz_convert("UTC")
    ts_utc_15 = ts_utc.dt.floor("15min")
    return ts_utc_15.dt.tz_convert(NY_TZ)

def make_key(df: pd.DataFrame) -> pd.Series:
    return (
        df[LANE_COL].astype("string").str.strip()
        + "||" + df[SID_COL].astype("string").str.strip()
        + "||" + df[TIME_COL].dt.strftime("%Y-%m-%d %H:%M:%S")
    )

def main():
    for p in [CRASH_PARQ, NONCRASH_PARQ, PRED_PARQ]:
        must_exist(p)

    print("Loading parquet files...")
    crash = pd.read_parquet(CRASH_PARQ)
    noncrash = pd.read_parquet(NONCRASH_PARQ)
    preds = pd.read_parquet(PRED_PARQ)

    # Drop missing timestamps (you had 10 in noncrash)
    crash = crash[crash[TIME_COL].notna()].copy()
    noncrash = noncrash[noncrash[TIME_COL].notna()].copy()
    preds = preds[preds[TIME_COL].notna()].copy()

    print("Normalizing timestamps to NY 15-min bins...")
    crash[TIME_COL] = floor15_ny(crash[TIME_COL])
    noncrash[TIME_COL] = floor15_ny(noncrash[TIME_COL])
    preds[TIME_COL] = floor15_ny(preds[TIME_COL])

    # Label
    crash["label"] = 1
    noncrash["label"] = 0

    # Combine crash + noncrash
    base = pd.concat([crash, noncrash], ignore_index=True)

    # Keys for join
    base["KEY"] = make_key(base)
    preds["KEY"] = make_key(preds)

    # --- Direct predictions at same timestamp (t -> pred t+30) ---
    preds_direct = preds[["KEY", "pred_speed_t_plus_30", "pred_volume_t_plus_30"]].copy()

    # --- Expected now: shift prediction forward by +30 minutes ---
    preds_shift = preds.copy()
    preds_shift[TIME_COL] = preds_shift[TIME_COL] + pd.Timedelta(minutes=30)
    preds_shift["KEY"] = (
        preds_shift[LANE_COL].astype("string").str.strip()
        + "||" + preds_shift[SID_COL].astype("string").str.strip()
        + "||" + preds_shift[TIME_COL].dt.strftime("%Y-%m-%d %H:%M:%S")
    )
    preds_shift = preds_shift[["KEY", "pred_speed_t_plus_30", "pred_volume_t_plus_30"]].rename(
        columns={
            "pred_speed_t_plus_30": "expected_speed_now",
            "pred_volume_t_plus_30": "expected_volume_now",
        }
    )

    print("Merging predictions...")
    out = base.merge(preds_direct, on="KEY", how="left")
    out = out.merge(preds_shift, on="KEY", how="left")

    # Deltas
    out["AVERAGE_SPEED"] = pd.to_numeric(out.get("AVERAGE_SPEED"), errors="coerce")
    out["TOTAL_VOLUME"] = pd.to_numeric(out.get("TOTAL_VOLUME"), errors="coerce")

    out["delta_speed_now"] = out["AVERAGE_SPEED"] - out["expected_speed_now"]
    out["delta_volume_now"] = out["TOTAL_VOLUME"] - out["expected_volume_now"]

    # Quick diagnostics
    print("\n=== DIAGNOSTICS ===")
    print("Rows:", len(out))
    print("Label counts:", out["label"].value_counts(dropna=False).to_dict())

    miss_direct = out["pred_speed_t_plus_30"].isna().mean()
    miss_expected = out["expected_speed_now"].isna().mean()
    print(f"Missing pred_t+30 rate: {miss_direct:.3%}")
    print(f"Missing expected_now rate: {miss_expected:.3%}")

    # Remove helper key
    out = out.drop(columns=["KEY"])

    print("\nWriting parquet:", OUT_PARQ)
    out.to_parquet(OUT_PARQ, index=False)

    # Excel: remove tz
    out_xlsx = out.copy()
    out_xlsx[TIME_COL] = pd.to_datetime(out_xlsx[TIME_COL], errors="coerce")
    if getattr(out_xlsx[TIME_COL].dt, "tz", None) is not None:
        out_xlsx[TIME_COL] = out_xlsx[TIME_COL].dt.tz_convert(NY_TZ).dt.tz_localize(None)

    print("Writing excel:", OUT_XLSX)
    out_xlsx.to_excel(OUT_XLSX, index=False)

    print("\n✅ Done.")
    print(" -", OUT_PARQ)
    print(" -", OUT_XLSX)

if __name__ == "__main__":
    main()
