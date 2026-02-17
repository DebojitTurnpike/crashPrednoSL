import os
import pandas as pd
import numpy as np

NY_TZ = "America/New_York"

# You are running from:
# C:\Users\d.biswas\Documents\AccidentPred\crashPred\dataPrep\Crash_NonCrash4Training
RUN_DIR = os.getcwd()
BASE_READY = os.path.join(RUN_DIR, "ready4TrainingData")

CRASH_XLSX = os.path.join(BASE_READY, "CrashData_WithWeather_2024_2025_FINAL.xlsx")
NONCRASH_XLSX = os.path.join(BASE_READY, "NonCrash_Sample10pct_2024_2025_CLEAN.xlsx")

CRASH_OUT_PARQUET = os.path.join(BASE_READY, "CrashData_WithWeather_2024_2025_FINAL.parquet")
NONCRASH_OUT_PARQUET = os.path.join(BASE_READY, "NonCrash_Sample10pct_2024_2025_CLEAN.parquet")

TIME_COL = "ROLLUP_TIMESTAMP"


def ensure_exists(p: str, label: str):
    if not os.path.exists(p):
        raise FileNotFoundError(f"{label} not found: {p}")


def parse_excel_mixed_datetime(series: pd.Series) -> pd.Series:
    """
    Parse mixed XLSX datetime column that may contain:
      - Python datetime objects
      - strings
      - Excel serial numbers
    Output: tz-aware America/New_York timestamps.
    """
    s = series.copy()

    # initialize output (tz-aware)
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    # detect numeric excel serials (must be actual numbers, not numeric strings)
    is_num = s.apply(lambda x: isinstance(x, (int, float, np.integer, np.floating)))

    if is_num.any():
        excel_epoch = pd.Timestamp("1899-12-30")  # Excel epoch
        dt_num = excel_epoch + pd.to_timedelta(pd.to_numeric(s[is_num], errors="coerce"), unit="D")
        out.loc[is_num] = dt_num

    # parse everything else as datetime
    is_other = ~is_num
    if is_other.any():
        dt_other = pd.to_datetime(s[is_other], errors="coerce")
        out.loc[is_other] = dt_other

    # Localize to NY safely (handles DST ambiguous/nonexistent)
    # If already tz-aware, just convert
    dt = pd.to_datetime(out, errors="coerce")

    # If tz-aware: convert to NY; if tz-naive: localize to NY
    try:
        if getattr(dt.dt, "tz", None) is not None:
            dt_ny = dt.dt.tz_convert(NY_TZ)
        else:
            dt_ny = dt.dt.tz_localize(NY_TZ, ambiguous="infer", nonexistent="shift_forward")
    except Exception:
        # Fallback: localize with ambiguous='NaT' if inference fails
        dt_ny = dt.dt.tz_localize(NY_TZ, ambiguous="NaT", nonexistent="shift_forward")

    return dt_ny


def normalize_rollup_15min(df: pd.DataFrame) -> pd.DataFrame:
    if TIME_COL not in df.columns:
        raise ValueError(f"Missing required column: {TIME_COL}")

    dt_ny = parse_excel_mixed_datetime(df[TIME_COL])

    # Floor safely in UTC then convert back to NY to avoid DST floor issues
    dt_utc = dt_ny.dt.tz_convert("UTC")
    dt_utc_15 = dt_utc.dt.floor("15min")
    dt_ny_15 = dt_utc_15.dt.tz_convert(NY_TZ)

    df = df.copy()
    df[TIME_COL] = dt_ny_15
    return df


def print_summary(name: str, df: pd.DataFrame):
    print("\n==============================")
    print(f"DATASET: {name}")
    print("Rows:", len(df))
    ok = df[TIME_COL].notna().sum()
    bad = len(df) - ok
    print(f"{TIME_COL} parsed OK: {ok} | Missing/bad: {bad}")

    if ok > 0:
        print("Min time:", df[TIME_COL].min())
        print("Max time:", df[TIME_COL].max())
        years = df[TIME_COL].dt.year.value_counts().sort_index()
        print("Year counts:", years.to_dict())


def main():
    os.makedirs(BASE_READY, exist_ok=True)

    ensure_exists(CRASH_XLSX, "Crash XLSX")
    ensure_exists(NONCRASH_XLSX, "NonCrash XLSX")

    print("Reading XLSX files...")
    crash = pd.read_excel(CRASH_XLSX, engine="openpyxl")
    noncrash = pd.read_excel(NONCRASH_XLSX, engine="openpyxl")

    print("Normalizing ROLLUP_TIMESTAMP to NY tz and 15-min bins...")
    crash_n = normalize_rollup_15min(crash)
    noncrash_n = normalize_rollup_15min(noncrash)

    print_summary("CRASH (normalized)", crash_n)
    print_summary("NONCRASH (normalized)", noncrash_n)

    print("\nWriting Parquet outputs...")
    crash_n.to_parquet(CRASH_OUT_PARQUET, index=False)
    noncrash_n.to_parquet(NONCRASH_OUT_PARQUET, index=False)

    print("✅ Wrote:")
    print(" -", CRASH_OUT_PARQUET)
    print(" -", NONCRASH_OUT_PARQUET)
    print("\nDone.")


if __name__ == "__main__":
    main()
