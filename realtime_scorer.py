# realtime_scorer.py
"""
Real-time Crash Risk Scorer (SunGuide + VisualCrossing + TimeSeries XGB + Crash XGB)

What it does (every run):
1) Calls SunGuide API for last N 15-min bins (default 8 => 2 hours)
2) Adds weather from VisualCrossing (optional; cached on disk)
3) Builds the SAME style of time-series features (lags/rollings/time encodings)
4) Runs xgb_speed + xgb_volume to get pred_speed_t_plus_30 / pred_volume_t_plus_30
5) Builds expected_now by shifting (t-30 pred) -> time t
6) Builds crash-model features and scores xgb_crash_classifier_strict_time
7) Outputs JSON + optionally writes results to MongoDB

Run:
  python realtime_scorer.py --bins 8 --write-mongo 1

Env needed (see .env template at bottom of this message):
  SUNGUIDE_API_BASE=http://10.231.0.13:8089
  VC_API_KEY=...
  MODEL_SPEED=/path/xgb_speed.json
  MODEL_VOLUME=/path/xgb_volume.json
  MODEL_CRASH=/path/xgb_crash_classifier_strict_time.json
  MONGO_URI=... (optional)
"""

import os
import json
import time
import math
import argparse
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests

import xgboost as xgb

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    ZoneInfo = None

NY_TZ = ZoneInfo("America/New_York") if ZoneInfo else None


from pathlib import Path

try:
    from dotenv import load_dotenv
    # Load .env from the same directory as this script (not from current terminal folder)
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path, override=True)
except Exception as e:
    print("Warning: dotenv not loaded:", e)


def _json_safe(obj):
    # pandas / numpy
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()

    # python datetime/date
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    return obj

# ----------------------------
# Helpers: time / bins
# ----------------------------
def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def floor_to_15min(dt: datetime) -> datetime:
    # dt timezone-aware recommended
    epoch = dt.timestamp()
    fifteen = 15 * 60
    floored = math.floor(epoch / fifteen) * fifteen
    return datetime.fromtimestamp(floored, tz=dt.tzinfo or timezone.utc)

def last_closed_bin_end_utc(now: datetime) -> datetime:
    # "closed" bin end is the most recent 15-min boundary strictly before now
    flo = floor_to_15min(now)
    if flo >= now:
        flo = flo - timedelta(minutes=15)
    return flo

def to_ny_aware(dt_utc: datetime) -> datetime:
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    return dt_utc.astimezone(NY_TZ) if NY_TZ else dt_utc

def to_utc_aware(dt_any: datetime) -> datetime:
    if dt_any.tzinfo is None:
        return dt_any.replace(tzinfo=timezone.utc)
    return dt_any.astimezone(timezone.utc)

def ensure_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)

def norm_to_ny_15min_aware(ts_utc_series: pd.Series) -> pd.Series:
    # Input: UTC-aware timestamps
    # Output: NY-aware timestamps floored to 15 min
    s = ensure_dt(ts_utc_series)
    if NY_TZ:
        s = s.dt.tz_convert("America/New_York")
    # Floor in local tz safely (pandas handles DST with tz-aware)
    s = s.dt.floor("15min")
    return s


# ----------------------------
# SunGuide API
# ----------------------------
def fetch_sunguide_latest(api_base: str, bins: int, timeout_s: int = 60) -> pd.DataFrame:
    url = f"{api_base.rstrip('/')}/api/sunguide/latest"
    r = requests.get(url, params={"bins": bins}, timeout=timeout_s)
    r.raise_for_status()
    js = r.json()
    data = js.get("data", [])
    df = pd.DataFrame(data)
    if df.empty:
        return df

    # SunGuide API returns ISO timestamps (usually UTC Z)
    df["ROLLUP_TIMESTAMP"] = ensure_dt(df["ROLLUP_TIMESTAMP"])
    # Normalize to NY 15-min bins for internal consistency
    df["ROLLUP_TIMESTAMP_NY"] = norm_to_ny_15min_aware(df["ROLLUP_TIMESTAMP"])
    return df


# ----------------------------
# VisualCrossing Weather (optional, cached)
# ----------------------------
def _cache_dir() -> str:
    d = os.path.join(os.getcwd(), "weather_cache")
    os.makedirs(d, exist_ok=True)
    return d

def _cache_key(lat: float, lon: float, day_ny: str) -> str:
    # Round to reduce duplicates (detectors move slightly)
    return f"{round(lat,5)}_{round(lon,5)}_{day_ny}.json"

def _load_cache(lat: float, lon: float, day_ny: str) -> Optional[dict]:
    path = os.path.join(_cache_dir(), _cache_key(lat, lon, day_ny))
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def _save_cache(lat: float, lon: float, day_ny: str, payload: dict) -> None:
    path = os.path.join(_cache_dir(), _cache_key(lat, lon, day_ny))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

def vc_fetch_day(lat: float, lon: float, day_ny: str, api_key: str, timeout_s: int = 60) -> dict:
    """
    Fetch 1 day hourly weather (VisualCrossing timeline endpoint).
    We cache per (lat,lon,day) to avoid hammering VC.
    """
    cached = _load_cache(lat, lon, day_ny)
    if cached is not None:
        return cached

    # VisualCrossing timeline:
    # https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/{day}
    # We request JSON and include hourly
    base = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    url = f"{base}/{lat},{lon}/{day_ny}"
    params = {
        "unitGroup": "us",
        "include": "hours",
        "key": api_key,
        "contentType": "json",
    }
    r = requests.get(url, params=params, timeout=timeout_s)
    r.raise_for_status()
    payload = r.json()
    _save_cache(lat, lon, day_ny, payload)
    return payload

def vc_join_weather(df: pd.DataFrame, api_key: str) -> pd.DataFrame:
    """
    Adds weather columns aligned to NY 15-min bins.
    Strategy: for each unique detector (lat,lon) and day, fetch hourly,
    then map each 15-min bin to the hour record.
    """
    if df.empty:
        return df

    needed_cols = ["Detector_LATITUDE", "Detector_LONGITUDE", "ROLLUP_TIMESTAMP_NY"]
    for c in needed_cols:
        if c not in df.columns:
            df[c] = np.nan

    df = df.copy()
    # If missing lat/lon, skip
    valid = df.dropna(subset=["Detector_LATITUDE", "Detector_LONGITUDE", "ROLLUP_TIMESTAMP_NY"])
    if valid.empty:
        return df

    # Prep weather fields
    weather_cols = [
        "TEMPERATURE", "FEELSLIKE", "HUMIDITY", "DEW", "PRECIP",
        "PRECIPPROB", "SOLARRADIATION", "WINDSPEED", "PRESSURE",
        "VISIBILITY", "CLOUDCOVER", "CONDITIONS"
    ]
    for c in weather_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Build map for (lat,lon,day)-> hourly dict
    groups = valid.groupby(
        [valid["Detector_LATITUDE"].round(5), valid["Detector_LONGITUDE"].round(5), valid["ROLLUP_TIMESTAMP_NY"].dt.strftime("%Y-%m-%d")]
    )

    # For each group, fetch once and apply
    for (lat_r, lon_r, day_ny), idx in groups.groups.items():
        lat = float(lat_r)
        lon = float(lon_r)
        payload = vc_fetch_day(lat, lon, day_ny, api_key=api_key)

        hours = payload.get("hours", [])
        if not hours:
            continue

        # Build hour lookup by hour integer (0-23)
        hour_map = {}
        for h in hours:
            # VisualCrossing hour has "datetime" like "13:00:00"
            hh = h.get("datetime", "")
            try:
                hour_int = int(str(hh).split(":")[0])
            except Exception:
                continue
            hour_map[hour_int] = h

        # Apply rows
        rows = df.loc[idx]
        hour_ints = rows["ROLLUP_TIMESTAMP_NY"].dt.hour.values
        for i, hr in zip(idx, hour_ints):
            rec = hour_map.get(int(hr))
            if not rec:
                continue
            df.at[i, "TEMPERATURE"] = rec.get("temp")
            df.at[i, "FEELSLIKE"] = rec.get("feelslike")
            df.at[i, "HUMIDITY"] = rec.get("humidity")
            df.at[i, "DEW"] = rec.get("dew")
            df.at[i, "PRECIP"] = rec.get("precip")
            df.at[i, "PRECIPPROB"] = rec.get("precipprob")
            df.at[i, "SOLARRADIATION"] = rec.get("solarradiation")
            df.at[i, "WINDSPEED"] = rec.get("windspeed")
            df.at[i, "PRESSURE"] = rec.get("pressure")
            df.at[i, "VISIBILITY"] = rec.get("visibility")
            df.at[i, "CLOUDCOVER"] = rec.get("cloudcover")
            df.at[i, "CONDITIONS"] = rec.get("conditions")

    return df


# ----------------------------
# Feature engineering (TS)
# ----------------------------
def conditions_to_code(s: str) -> int:
    if not isinstance(s, str) or not s.strip():
        return 0
    t = s.strip().lower()
    # very coarse mapping (same idea you used in FE)
    if "rain" in t or "drizzle" in t or "shower" in t:
        return 2
    if "storm" in t or "thunder" in t:
        return 3
    if "fog" in t or "mist" in t or "haze" in t:
        return 4
    if "snow" in t or "ice" in t or "sleet" in t:
        return 5
    if "cloud" in t or "overcast" in t:
        return 1
    return 0  # clear/unknown

def add_time_encodings(df: pd.DataFrame, ts_col: str = "ROLLUP_TIMESTAMP_NY") -> pd.DataFrame:
    df = df.copy()
    ts = df[ts_col]
    df["year"] = ts.dt.year.astype("int16")
    df["month"] = ts.dt.month.astype("int8")
    df["day"] = ts.dt.day.astype("int8")
    df["hour"] = ts.dt.hour.astype("int8")
    df["minute"] = ts.dt.minute.astype("int8")
    df["dow"] = ts.dt.dayofweek.astype("int8")
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")

    # cyclical
    hour = df["hour"].astype(float)
    dow = df["dow"].astype(float)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    # extra fields that your TS model previously required
    df["ny_hour"] = df["hour"].astype("int8")
    df["ny_dow"] = df["dow"].astype("int8")
    df["ny_month"] = df["month"].astype("int8")
    df["minute_of_day"] = (df["hour"].astype(int) * 60 + df["minute"].astype(int)).astype("int16")
    return df

def add_lags_and_rollings(
    df: pd.DataFrame,
    group_cols: List[str],
    speed_col: str = "AVERAGE_SPEED",
    vol_col: str = "TOTAL_VOLUME",
    weather_cols: Optional[List[str]] = None,
    ts_col: str = "ROLLUP_TIMESTAMP_NY"
) -> pd.DataFrame:
    df = df.sort_values(group_cols + [ts_col]).copy()
    g = df.groupby(group_cols, sort=False)

    # lags for traffic
    for k in [1, 2, 3, 4, 6, 8]:
        df[f"{vol_col}_lag{k}"] = g[vol_col].shift(k)
        df[f"{speed_col}_lag{k}"] = g[speed_col].shift(k)

    # rolling windows: 4/8/16 bins
    for w in [4, 8, 16]:
        df[f"{vol_col}_roll{w}_mean"] = g[vol_col].rolling(w).mean().reset_index(level=group_cols, drop=True)
        df[f"{vol_col}_roll{w}_std"]  = g[vol_col].rolling(w).std().reset_index(level=group_cols, drop=True)
        df[f"{vol_col}_roll{w}_min"]  = g[vol_col].rolling(w).min().reset_index(level=group_cols, drop=True)
        df[f"{vol_col}_roll{w}_max"]  = g[vol_col].rolling(w).max().reset_index(level=group_cols, drop=True)

        df[f"{speed_col}_roll{w}_mean"] = g[speed_col].rolling(w).mean().reset_index(level=group_cols, drop=True)
        df[f"{speed_col}_roll{w}_std"]  = g[speed_col].rolling(w).std().reset_index(level=group_cols, drop=True)
        df[f"{speed_col}_roll{w}_min"]  = g[speed_col].rolling(w).min().reset_index(level=group_cols, drop=True)
        df[f"{speed_col}_roll{w}_max"]  = g[speed_col].rolling(w).max().reset_index(level=group_cols, drop=True)

    # diffs
    df["SPEED_diff1"] = g[speed_col].diff(1)
    df["VOL_diff1"] = g[vol_col].diff(1)

    # lags for weather (lightweight)
    if weather_cols:
        for c in weather_cols:
            for k in [1, 2, 4]:
                df[f"{c}_lag{k}"] = g[c].shift(k)

    return df

def build_timeseries_feature_frame(df_hist: pd.DataFrame) -> pd.DataFrame:
    """
    df_hist expected columns:
      LANE_NAME, SID, ROLLUP_TIMESTAMP (UTC), ROLLUP_TIMESTAMP_NY (NY tz), AVERAGE_SPEED, TOTAL_VOLUME
      + weather cols optional
    """
    df = df_hist.copy()
    if "CONDITIONS" in df.columns:
        df["CONDITIONS_CODE"] = df["CONDITIONS"].apply(conditions_to_code).astype("int16")
    else:
        df["CONDITIONS_CODE"] = 0

    df = add_time_encodings(df, ts_col="ROLLUP_TIMESTAMP_NY")

    weather_base_cols = [
        "TEMPERATURE","FEELSLIKE","HUMIDITY","DEW","PRECIP","PRECIPPROB",
        "SOLARRADIATION","WINDSPEED","PRESSURE","VISIBILITY","CLOUDCOVER"
    ]
    for c in weather_base_cols:
        if c not in df.columns:
            df[c] = np.nan

    df = add_lags_and_rollings(
        df,
        group_cols=["LANE_NAME","SID"],
        speed_col="AVERAGE_SPEED",
        vol_col="TOTAL_VOLUME",
        weather_cols=weather_base_cols,
        ts_col="ROLLUP_TIMESTAMP_NY"
    )
    return df


# ----------------------------
# XGBoost model utilities
# ----------------------------
def load_booster(path: str) -> xgb.Booster:
    b = xgb.Booster()
    b.load_model(path)
    return b

def booster_feature_names(b: xgb.Booster) -> List[str]:
    # xgboost json models typically store feature names
    fn = b.feature_names
    if fn is None:
        # Fallback: if none, we cannot safely infer order; force error
        raise ValueError("Model has no feature_names stored. Re-save with feature names, or provide a feature list.")
    return list(fn)

def dmatrix_from_df(df: pd.DataFrame, feature_names: List[str]) -> xgb.DMatrix:
    X = df.copy()
    for f in feature_names:
        if f not in X.columns:
            X[f] = 0.0
    X = X[feature_names]

    # Make numeric
    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    # XGBoost can handle NaNs; keep them
    return xgb.DMatrix(X.values, feature_names=feature_names)

def predict_booster(b: xgb.Booster, df: pd.DataFrame) -> np.ndarray:
    fn = booster_feature_names(b)
    dmat = dmatrix_from_df(df, fn)
    return b.predict(dmat)


# ----------------------------
# Expected-now (shift) + crash features
# ----------------------------
def add_expected_now_from_shift(df: pd.DataFrame) -> pd.DataFrame:
    """
    expected_speed_now(t) = pred_speed_t_plus_30(t-30) shifted forward
    """
    df = df.copy()
    df["ts_key"] = df["ROLLUP_TIMESTAMP_NY"]

    shifted = df[["LANE_NAME","SID","ts_key","pred_speed_t_plus_30","pred_volume_t_plus_30"]].copy()
    shifted["ts_key"] = shifted["ts_key"] + pd.Timedelta(minutes=30)
    shifted = shifted.rename(columns={
        "pred_speed_t_plus_30": "expected_speed_now",
        "pred_volume_t_plus_30": "expected_volume_now"
    })

    out = df.merge(
        shifted,
        on=["LANE_NAME","SID","ts_key"],
        how="left"
    )

    # deltas
    out["delta_speed_obs_minus_expected"] = out["AVERAGE_SPEED"] - out["expected_speed_now"]
    out["delta_vol_obs_minus_expected"] = out["TOTAL_VOLUME"] - out["expected_volume_now"]

    return out.drop(columns=["ts_key"], errors="ignore")

def add_observed_change_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["LANE_NAME","SID","ROLLUP_TIMESTAMP_NY"]).copy()
    g = df.groupby(["LANE_NAME","SID"], sort=False)

    out = df.copy()
    out["speed_change_15"] = g["AVERAGE_SPEED"].diff(1)
    out["volume_change_15"] = g["TOTAL_VOLUME"].diff(1)

    # rolling std over last 30 min (2 bins)
    out["speed_std_30"] = g["AVERAGE_SPEED"].rolling(2).std().reset_index(level=[0,1], drop=True)
    out["vol_std_30"] = g["TOTAL_VOLUME"].rolling(2).std().reset_index(level=[0,1], drop=True)

    return out

def build_crash_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    IMPORTANT:
    We do NOT hardcode the 21 features. We dynamically build whatever the crash model expects
    using its stored feature_names, and fill missing columns with 0.
    """
    df = df.copy()
    # Add time encodings (the crash model often includes these)
    df = add_time_encodings(df, ts_col="ROLLUP_TIMESTAMP_NY")

    # conditions_code again (if needed)
    if "CONDITIONS" in df.columns and "CONDITIONS_CODE" not in df.columns:
        df["CONDITIONS_CODE"] = df["CONDITIONS"].apply(conditions_to_code).astype("int16")
    if "CONDITIONS_CODE" not in df.columns:
        df["CONDITIONS_CODE"] = 0

    # Provide the core signals we trained on
    # (names are generic; the model will pick what it needs by feature_names)
    # Observed + predicted
    # - AVERAGE_SPEED, TOTAL_VOLUME
    # - pred_speed_t_plus_30, pred_volume_t_plus_30
    # - expected_speed_now, expected_volume_now
    # - deltas + short-term change
    keep = df
    return keep


# ----------------------------
# MongoDB writer (optional)
# ----------------------------
def write_to_mongo(records: List[dict], mongo_uri: str, mongo_db: str, mongo_col: str) -> None:
    from pymongo import MongoClient
    if not records:
        return
    client = MongoClient(mongo_uri)
    col = client[mongo_db][mongo_col]
    # Upsert by lane+sid+timestamp to avoid duplicates
    ops = []
    from pymongo import UpdateOne
    for r in records:
        key = {
            "LANE_NAME": r.get("LANE_NAME"),
            "SID": r.get("SID"),
            "ROLLUP_TIMESTAMP_NY": r.get("ROLLUP_TIMESTAMP_NY"),
        }
        ops.append(UpdateOne(key, {"$set": r}, upsert=True))
    if ops:
        col.bulk_write(ops, ordered=False)


# ----------------------------
# Main pipeline
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", type=int, default=int(os.getenv("BINS", "8")))
    ap.add_argument("--threshold", type=float, default=float(os.getenv("CRASH_THRESHOLD", "0.60")))
    ap.add_argument("--write-mongo", type=int, default=int(os.getenv("WRITE_MONGO", "0")))
    ap.add_argument("--out-json", type=str, default=os.getenv("OUT_JSON", "latest_crash_risk.json"))
    args = ap.parse_args()

    api_base = os.getenv("SUNGUIDE_API_BASE", "http://localhost:8089")
    vc_key = os.getenv("VC_API_KEY", "").strip()

    model_speed_path = os.getenv("MODEL_SPEED", "").strip()
    model_vol_path = os.getenv("MODEL_VOLUME", "").strip()
    model_crash_path = os.getenv("MODEL_CRASH", "").strip()

    if not model_speed_path or not model_vol_path or not model_crash_path:
        raise SystemExit("Missing MODEL_SPEED / MODEL_VOLUME / MODEL_CRASH env vars (paths to .json models).")

    print(f"SunGuide API: {api_base}")
    print(f"bins={args.bins} threshold={args.threshold}")

    # 1) Pull live traffic
    df = fetch_sunguide_latest(api_base, bins=args.bins)
    if df.empty:
        print("No data returned from SunGuide.")
        return

    # 2) Weather (optional)
    if vc_key:
        print("Joining VisualCrossing weather (cached)...")
        df = vc_join_weather(df, vc_key)
    else:
        print("VC_API_KEY not set -> skipping weather (will be NaN/0).")

    # 3) Build TS features + predict t+30
    print("Building time-series features...")
    feats = build_timeseries_feature_frame(df)

    print("Loading time-series models...")
    ms = load_booster(model_speed_path)
    mv = load_booster(model_vol_path)

    # predict for ALL rows we have (bins window)
    print("Predicting speed/volume t+30...")
    feats["pred_speed_t_plus_30"] = predict_booster(ms, feats).astype("float32")
    feats["pred_volume_t_plus_30"] = predict_booster(mv, feats).astype("float32")

    # 4) expected_now from shift + observed changes
    feats = add_expected_now_from_shift(feats)
    feats = add_observed_change_features(feats)

    # 5) Score crash model for the LATEST closed bin only
    latest_ts = feats["ROLLUP_TIMESTAMP_NY"].max()
    latest = feats[feats["ROLLUP_TIMESTAMP_NY"] == latest_ts].copy()

    print(f"Scoring crash risk for latest NY bin: {latest_ts} | rows={len(latest)}")

    crash_model = load_booster(model_crash_path)
    crash_feats = build_crash_feature_frame(latest)

    # Crash probability
    latest["crash_probability"] = predict_booster(crash_model, crash_feats).astype("float32")
    latest["alert"] = (latest["crash_probability"] >= args.threshold)

    def band(p: float) -> str:
        if p >= 0.70: return "HIGH"
        if p >= 0.60: return "MEDIUM"
        if p >= 0.45: return "LOW"
        return "MIN"

    latest["risk_band"] = latest["crash_probability"].apply(lambda x: band(float(x)))

    # Prepare output
    out_cols = [
        "ROLLUP_TIMESTAMP", "ROLLUP_TIMESTAMP_NY",
        "LANE_NAME", "SID", "SEGMENTS",
        "Detector_LATITUDE", "Detector_LONGITUDE",
        "TOTAL_VOLUME", "AVERAGE_SPEED",
        "pred_speed_t_plus_30", "pred_volume_t_plus_30",
        "expected_speed_now", "expected_volume_now",
        "delta_speed_obs_minus_expected", "delta_vol_obs_minus_expected",
        "speed_change_15", "volume_change_15", "speed_std_30", "vol_std_30",
        "TEMPERATURE", "PRECIP", "VISIBILITY", "WINDSPEED", "CONDITIONS_CODE",
        "crash_probability", "risk_band", "alert",
    ]
    for c in out_cols:
        if c not in latest.columns:
            latest[c] = np.nan

    # JSON-friendly types
    out_df = latest[out_cols].copy()
    out_df["ROLLUP_TIMESTAMP"] = pd.to_datetime(out_df["ROLLUP_TIMESTAMP"], utc=True, errors="coerce")
    # Store NY timestamp as ISO string (timezone-aware)
    out_df["ROLLUP_TIMESTAMP_NY"] = out_df["ROLLUP_TIMESTAMP_NY"].astype(str)

    records = out_df.replace({np.nan: None}).to_dict(orient="records")

    # Write local JSON
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(
    {"ok": True, "timestamp_ny": str(latest_ts), "rows": len(records), "data": records},
    f,
    indent=2,
    default=_json_safe
)

    print(f"✅ Wrote: {args.out_json}")

    # 6) Optional MongoDB write
    if args.write_mongo == 1:
        mongo_uri = os.getenv("MONGO_URI", "").strip()
        mongo_db = os.getenv("MONGO_DB", "trafficData").strip()
        mongo_col = os.getenv("MONGO_COLLECTION", "crash_risk_live").strip()
        if not mongo_uri:
            print("WRITE_MONGO=1 but MONGO_URI not set. Skipping Mongo.")
        else:
            # store timestamp string for upsert key
            for r in records:
                r["ROLLUP_TIMESTAMP_NY"] = r["ROLLUP_TIMESTAMP_NY"]
            write_to_mongo(records, mongo_uri, mongo_db, mongo_col)
            print(f"✅ Upserted {len(records)} rows to MongoDB: {mongo_db}.{mongo_col}")

    # Print quick summary
    alerts = [r for r in records if r.get("alert")]
    print(f"Alerts this bin: {len(alerts)} / {len(records)}")
    if alerts:
        top = sorted(alerts, key=lambda x: (x.get("crash_probability") or 0), reverse=True)[:10]
        print("Top alerts (lane, sid, p):")
        for a in top:
            print(f"  {a['LANE_NAME']} | {a['SID']} | p={a['crash_probability']:.3f} | {a['risk_band']}")

if __name__ == "__main__":
    main()
