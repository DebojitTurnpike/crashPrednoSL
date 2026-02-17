/**
 * crashRiskApi.js (corridor-aware, management-friendly explanations)
 *
 * Reads lane-level crash risk from MongoDB (crash_risk_live) and provides:
 *  - latest lane snapshot
 *  - latest corridor alerts (using SEGMENTS as corridor_id)
 *  - corridor alerts over a time range
 *  - latest snapshot for a corridor
 *
 * IMPORTANT:
 *  Your Mongo schema (from your sample) uses:
 *    - crash_probability (float)
 *    - alert (bool)
 *    - SEGMENTS (string, e.g., "RR 3")
 *    - ROLLUP_TIMESTAMP (ISODate)
 *
 * Explanations / "top 3 reasons":
 *  - Preferred: scorer writes a field `top_reasons` (array of 3 objects)
 *      top_reasons: [
 *        { feature: "delta_speed_obs_minus_expected", value: -12.3, contribution: 0.08 },
 *        { feature: "speed_change_15", value: -10, contribution: 0.05 },
 *        { feature: "vol_std_30", value: 55.2, contribution: 0.03 }
 *      ]
 *    (contribution can be SHAP, gain-based, or any normalized importance you choose)
 *
 *  - Fallback: if top_reasons does not exist, API derives a simple explanation
 *    from existing numeric fields.
 */

"use strict";

const express = require("express");
const { MongoClient } = require("mongodb");
const helmet = require("helmet");
const compression = require("compression");
const cors = require("cors");
const morgan = require("morgan");
const { z } = require("zod");
const fs = require("fs");
const path = require("path");

let dotenvLoaded = false;
try {
  require("dotenv").config({ path: path.join(__dirname, ".env") });
  dotenvLoaded = true;
} catch {
  // optional
}

// Env
// Health status file written by realtime_scorer.py
const HEALTH_STATUS_PATH =
    process.env.HEALTH_STATUS_PATH ||
    path.join(__dirname, "health_status.json"); // adjust if API runs elsewhere

console.log("[ENV] dotenvLoaded:", dotenvLoaded);
console.log("[ENV] cwd:", process.cwd());
console.log("[ENV] HEALTH_STATUS_PATH:", HEALTH_STATUS_PATH);


const API_PORT = parseInt(process.env.API_PORT || "8095", 10);
const MONGO_URI = process.env.MONGO_URI || "mongodb://localhost:27017";
const MONGO_DB = process.env.MONGO_DB || "trafficData";
const MONGO_COLLECTION = process.env.MONGO_COLLECTION || "crash_risk_live";

const DEFAULT_LOOKBACK_DAYS = parseInt(process.env.DEFAULT_LOOKBACK_DAYS || "2", 10);
const DEFAULT_LIMIT = Math.min(parseInt(process.env.DEFAULT_LIMIT || "5000", 10), 20000);
const CORS_ORIGIN = process.env.CORS_ORIGIN || "*";

// Field names (match your Mongo docs)
const FIELD_TS = "ROLLUP_TIMESTAMP";
const FIELD_SID = "SID";
const FIELD_LANE = "LANE_NAME";
const FIELD_SEGMENTS = "SEGMENTS";
const FIELD_PROB = process.env.PROB_FIELD || "crash_probability";
const FIELD_ALERT = process.env.ALERT_FIELD || "alert";
const FIELD_REASONS = process.env.REASONS_FIELD || "top_reasons"; // recommended to add in scorer

function readHealthStatusSafe() {
    try {
        if (!fs.existsSync(HEALTH_STATUS_PATH)) {
            return {
                ok: false,
                error: "health_status.json not found",
                path: HEALTH_STATUS_PATH,
            };
        }
        const raw = fs.readFileSync(HEALTH_STATUS_PATH, "utf-8");
        const parsed = JSON.parse(raw);
        return { ok: true, ...parsed, path: HEALTH_STATUS_PATH };
    } catch (e) {
        return {
            ok: false,
            error: "failed to read/parse health_status.json",
            message: String(e?.message || e),
            path: HEALTH_STATUS_PATH,
        };
    }
}

function clampInt(n, min, max) {
    if (!Number.isFinite(n)) return min;
    return Math.max(min, Math.min(max, Math.trunc(n)));
}

function toNumberOrNull(v) {
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
}

function safeCorridorId(doc) {
    const v = doc?.[FIELD_SEGMENTS];
    if (v === null || v === undefined) return "UNMAPPED";
    const s = String(v).trim();
    return s.length ? s : "UNMAPPED";
}

function sanitizeTopReasons(
    topReasons,
    { onlyPositiveForAlerts = false, isAlert = false } = {}
) {
    if (!Array.isArray(topReasons)) return [];

    const blocked = new Set([
        "Detector_LATITUDE",
        "Detector_LONGITUDE",
        "LANE_NAME",
        "SID",
        "SEGMENTS",
        "ROLLUP_TIMESTAMP",
        "ROLLUP_TIMESTAMP_NY",
        "CONDITIONS_CODE",
        "year", "month", "day", "hour", "minute", "dow", "is_weekend",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "ny_hour", "ny_dow", "ny_month", "minute_of_day",
    ]);

    const cleaned = topReasons
        .filter(r => r && typeof r.feature === "string")
        .filter(r => !blocked.has(r.feature))
        .map(r => {
            const contrib = Number(r.contribution ?? r.shap ?? r.weight);
            const valNum = (r.value === null || r.value === undefined) ? null : (Number.isFinite(Number(r.value)) ? Number(r.value) : r.value);
            return { feature: r.feature, value: valNum, contribution: contrib };
        })
        .filter(r => Number.isFinite(r.contribution));

    if (cleaned.length === 0) return [];

    if (isAlert && onlyPositiveForAlerts) {
        const pos = cleaned.filter(r => r.contribution > 0).sort((a, b) => b.contribution - a.contribution);
        if (pos.length) return pos.slice(0, 3);
    }

    cleaned.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
    return cleaned.slice(0, 3);
}


function humanizeFeatureName(f) {
    const map = {
        delta_speed_obs_minus_expected: "Speed vs expected (delta)",
        delta_vol_obs_minus_expected: "Volume vs expected (delta)",
        speed_change_15: "Speed change (15 min)",
        volume_change_15: "Volume change (15 min)",
        speed_std_30: "Speed volatility (30 min)",
        vol_std_30: "Volume volatility (30 min)",
        pred_speed_t_plus_30: "Predicted speed (t+30)",
        pred_volume_t_plus_30: "Predicted volume (t+30)",
        WINDSPEED: "Wind speed",
        PRECIP: "Precipitation",
        VISIBILITY: "Visibility",
        HUMIDITY: "Humidity"
    };
    return map[f] || f;
}

/**
 * Preferred: pass-through scorer-provided reasons if present and well-formed.
 * Fallback: create a concise 3-item explanation using existing fields.
 */
function buildTopReasons(doc) {
    const isAlert = Boolean(doc?.[FIELD_ALERT]);

    // 1) If scorer provided reasons, sanitize + label them
    const reasons = doc?.[FIELD_REASONS];
    if (Array.isArray(reasons) && reasons.length) {
        const sanitized = sanitizeTopReasons(reasons, { isAlert });

        return sanitized.map((r) => ({
            feature: r.feature,
            label: humanizeFeatureName(r.feature),
            value: r.value ?? null,
            contribution: r.contribution ?? null,
        }));
    }

    // 2) Fallback heuristic reasons (no model explainability available)
    const candidates = [];

    const dSpeed = toNumberOrNull(doc?.delta_speed_obs_minus_expected);
    if (dSpeed !== null) {
        candidates.push({
            feature: "delta_speed_obs_minus_expected",
            value: dSpeed,
            note: dSpeed < 0 ? "Observed speed lower than expected" : "Observed speed higher than expected",
            score: Math.abs(dSpeed),
        });
    }

    const dVol = toNumberOrNull(doc?.delta_vol_obs_minus_expected);
    if (dVol !== null) {
        candidates.push({
            feature: "delta_vol_obs_minus_expected",
            value: dVol,
            note: dVol > 0 ? "Observed volume higher than expected" : "Observed volume lower than expected",
            score: Math.abs(dVol) / 10.0,
        });
    }

    const sChg = toNumberOrNull(doc?.speed_change_15);
    if (sChg !== null) {
        candidates.push({
            feature: "speed_change_15",
            value: sChg,
            note: sChg < 0 ? "Speed dropped in last 15 minutes" : "Speed increased in last 15 minutes",
            score: Math.abs(sChg) * 2.0,
        });
    }

    const vChg = toNumberOrNull(doc?.volume_change_15);
    if (vChg !== null) {
        candidates.push({
            feature: "volume_change_15",
            value: vChg,
            note: vChg < 0 ? "Volume dropped in last 15 minutes" : "Volume increased in last 15 minutes",
            score: Math.abs(vChg),
        });
    }

    const sStd = toNumberOrNull(doc?.speed_std_30);
    if (sStd !== null) {
        candidates.push({
            feature: "speed_std_30",
            value: sStd,
            note: "Speed variability over last 30 minutes",
            score: Math.abs(sStd),
        });
    }

    const vStd = toNumberOrNull(doc?.vol_std_30);
    if (vStd !== null) {
        candidates.push({
            feature: "vol_std_30",
            value: vStd,
            note: "Volume variability over last 30 minutes",
            score: Math.abs(vStd) / 5.0,
        });
    }

    candidates.sort((a, b) => b.score - a.score);
    const top = candidates.slice(0, 3);

    return top.map((t) => ({
        feature: t.feature,
        label: humanizeFeatureName(t.feature),
        value: t.value,
        contribution: null,
        note: t.note,
    }));
}


// App
const app = express();
app.use(helmet());
app.use(compression());
app.use(express.json({ limit: "1mb" }));
app.use(cors({ origin: CORS_ORIGIN === "*" ? true : CORS_ORIGIN }));
app.use(morgan("combined"));

// Mongo
const client = new MongoClient(MONGO_URI, {
    maxPoolSize: 20,
    serverSelectionTimeoutMS: 8000,
});

let coll = null;

async function connectMongo() {
    await client.connect();
    const db = client.db(MONGO_DB);
    coll = db.collection(MONGO_COLLECTION);

    // Helpful indexes
    await coll.createIndex({ [FIELD_TS]: -1 });
    await coll.createIndex({ [FIELD_TS]: -1, [FIELD_PROB]: -1 });
    await coll.createIndex({ [FIELD_TS]: -1, [FIELD_SEGMENTS]: 1 });
    await coll.createIndex({ [FIELD_SID]: 1, [FIELD_LANE]: 1, [FIELD_TS]: -1 });
}

function mustHaveCollection(req, res, next) {
    if (!coll) return res.status(503).json({ ok: false, error: "Mongo not ready" });
    next();
}

async function getLatestTimestamp() {
    const doc = await coll
        .find({}, { projection: { [FIELD_TS]: 1 } })
        .sort({ [FIELD_TS]: -1 })
        .limit(1)
        .next();
    return doc?.[FIELD_TS] || null;
}

// Health
app.get("/health", (_req, res) => {
    res.json({
        ok: true,
        mongoConnected: !!coll,
        dotenvLoaded,
        db: MONGO_DB,
        collection: MONGO_COLLECTION,
        prob_field: FIELD_PROB,
        alert_field: FIELD_ALERT,
        reasons_field: FIELD_REASONS,
        time: new Date().toISOString(),
    });
});
// Frontend-ready status of both data sources (SunGuide + VisualCrossing)
app.get("/api/system/data-sources", async (_req, res) => {
  try {
    const health = readHealthStatusSafe();
    const mongoLatestTs = coll ? await getLatestTimestamp() : null;

    const vc = health?.visualcrossing || health?.weather || null;

    res.json({
      ok: true,
      sources: {
        sunguide: health?.sunguide || null,
        visualcrossing: vc,
      },
      overall_ok: health?.overall_ok ?? false,
      status_color: health?.status_color ?? "red",
      latest_bin_ny: health?.latest_bin_ny ?? null,
      timestamp_utc: health?.timestamp_utc ?? null,
      mongo_latest_timestamp: mongoLatestTs || null,
      health_file: {
        ok: health?.ok ?? false,
        path: health?.path ?? HEALTH_STATUS_PATH,
        error: health?.error ?? null,
        message: health?.message ?? null,
      },
    });
  } catch (err) {
    res.status(500).json({ ok: false, error: String(err?.message || err) });
  }
});


// Latest timestamp
app.get("/api/crash-risk/latest-timestamp", mustHaveCollection, async (_req, res) => {
    try {
        const ts = await getLatestTimestamp();
        res.json({ ok: true, latest_timestamp: ts });
    } catch (err) {
        res.status(500).json({ ok: false, error: String(err?.message || err) });
    }
});

/**
 * Latest lane-level snapshot
 * Query:
 *   limit (default DEFAULT_LIMIT, max 20000)
 *   min_prob optional
 *   alerts_only optional (true/1)
 */
app.get("/api/crash-risk/latest", mustHaveCollection, async (req, res) => {
  try {
    const q = z
      .object({
        limit: z.string().optional(),
        min_prob: z.string().optional(),
        alerts_only: z.string().optional(),
      })
      .parse(req.query);

    const limit = clampInt(parseInt(q.limit || String(DEFAULT_LIMIT), 10), 1, 20000);
    const minProb = q.min_prob ? Number(q.min_prob) : null;
    const alertsOnly = q.alerts_only === "1" || q.alerts_only === "true";

    const latestTs = await getLatestTimestamp();
    if (!latestTs) return res.json({ ok: true, timestamp: null, count: 0, rows: [], data_sources: null });

    const filter = { [FIELD_TS]: latestTs };
    if (minProb !== null && Number.isFinite(minProb)) filter[FIELD_PROB] = { $gte: minProb };
    if (alertsOnly) filter[FIELD_ALERT] = true;

    const rows = await coll.find(filter).limit(limit).toArray();

    const enriched = rows.map((d) => ({
      ...d,
      corridor_id: safeCorridorId(d),
      top_reasons_raw: d?.[FIELD_REASONS] ?? null,
      top_reasons_resolved: buildTopReasons(d),
    }));

    const health = readHealthStatusSafe();
    const vc = health?.visualcrossing || health?.weather || null;
    const mongoLatestTs = await getLatestTimestamp(); // safe here

    res.json({
      ok: true,
      timestamp: latestTs,
      count: enriched.length,
      rows: enriched,
      data_sources: {
        overall_ok: health?.overall_ok ?? false,
        status_color: health?.status_color ?? "red",
        sunguide: health?.sunguide || null,
        visualcrossing: vc,
        latest_bin_ny: health?.latest_bin_ny ?? null,
        timestamp_utc: health?.timestamp_utc ?? null,
        mongo_latest_timestamp: mongoLatestTs || null,
      },
    });
  } catch (err) {
    res.status(400).json({ ok: false, error: String(err?.message || err) });
  }
});


/**
 * Latest corridor alerts (corridor_id = SEGMENTS)
 * Query:
 *   thr (default 0.6)
 *   topk (default 3)  -> top lanes to include
 *   min_lanes (default 1) -> number of lanes in corridor above thr to alert
 *   include_all (optional true/1) -> return all corridors, not only alerts
 */
app.get("/api/corridors/alerts/latest", mustHaveCollection, async (req, res) => {
    try {
        const q = z
            .object({
                thr: z.string().optional(),
                topk: z.string().optional(),
                min_lanes: z.string().optional(),
                include_all: z.string().optional(),
            })
            .parse(req.query);

        const thr = q.thr ? Number(q.thr) : 0.6;
        const topk = clampInt(parseInt(q.topk || "3", 10), 1, 20);
        const minLanes = clampInt(parseInt(q.min_lanes || "1", 10), 1, 50);
        const includeAll = q.include_all === "1" || q.include_all === "true";

        const latestTs = await getLatestTimestamp();
        if (!latestTs) return res.json({ ok: true, timestamp: null, corridors: [] });

        const rows = await coll
            .find({ [FIELD_TS]: latestTs }, { projection: { _id: 0 } })
            .toArray();

        const byCorr = new Map();

        for (const r of rows) {
            const prob = toNumberOrNull(r[FIELD_PROB]);
            if (prob === null) continue;

            const corridorId = safeCorridorId(r);
            if (!byCorr.has(corridorId)) {
                byCorr.set(corridorId, {
                    corridor_id: corridorId,
                    timestamp: latestTs,
                    corridor_risk: 0,
                    lanes_above_thr: 0,
                    lane_count: 0,
                    top_lanes: [],
                });
            }

            const agg = byCorr.get(corridorId);
            agg.lane_count += 1;
            if (prob > agg.corridor_risk) agg.corridor_risk = prob;
            if (prob >= thr) agg.lanes_above_thr += 1;

            agg.top_lanes.push({
                SID: r[FIELD_SID],
                LANE_NAME: r[FIELD_LANE],
                crash_probability: prob,
                alert: Boolean(r[FIELD_ALERT]),
                Detector_LATITUDE: r.Detector_LATITUDE,
                Detector_LONGITUDE: r.Detector_LONGITUDE,
                // attach reasons per lane
                top_reasons_resolved: buildTopReasons(r),
            });
        }

        let corridors = Array.from(byCorr.values()).map((c) => {
            c.top_lanes.sort((a, b) => b.crash_probability - a.crash_probability);
            c.top_lanes = c.top_lanes.slice(0, topk);
            c.is_alert = c.lanes_above_thr >= minLanes && c.corridor_risk >= thr;
            return c;
        });

        corridors.sort((a, b) => b.corridor_risk - a.corridor_risk);

        if (!includeAll) corridors = corridors.filter((c) => c.is_alert);

        res.json({
            ok: true,
            timestamp: latestTs,
            thr,
            topk,
            min_lanes: minLanes,
            include_all: includeAll,
            count: corridors.length,
            corridors,
        });
    } catch (err) {
        res.status(400).json({ ok: false, error: String(err?.message || err) });
    }
});

/**
 * Corridor alerts in a time range (grouped by bin)
 * Query:
 *   from=<ISO> optional (default now - DEFAULT_LOOKBACK_DAYS)
 *   to=<ISO> optional (default now)
 *   thr=0.6 optional
 *   topk=3 optional
 *   min_lanes=1 optional
 *   limit_bins=200 optional (max bins)
 */
app.get("/api/corridors/alerts", mustHaveCollection, async (req, res) => {
    try {
        const now = new Date();
        const defaultFrom = new Date(now.getTime() - DEFAULT_LOOKBACK_DAYS * 24 * 60 * 60 * 1000);

        const q = z
            .object({
                from: z.string().optional(),
                to: z.string().optional(),
                thr: z.string().optional(),
                topk: z.string().optional(),
                min_lanes: z.string().optional(),
                limit_bins: z.string().optional(),
            })
            .parse(req.query);

        const from = q.from ? new Date(q.from) : defaultFrom;
        const to = q.to ? new Date(q.to) : now;
        if (Number.isNaN(from.getTime()) || Number.isNaN(to.getTime())) {
            return res.status(400).json({ ok: false, error: "Invalid from/to datetime" });
        }

        const thr = q.thr ? Number(q.thr) : 0.6;
        const topk = clampInt(parseInt(q.topk || "3", 10), 1, 20);
        const minLanes = clampInt(parseInt(q.min_lanes || "1", 10), 1, 50);
        const limitBins = clampInt(parseInt(q.limit_bins || "200", 10), 1, 2000);

        // distinct timestamps in range
        const tsList = await coll.distinct(FIELD_TS, { [FIELD_TS]: { $gte: from, $lte: to } });
        tsList.sort((a, b) => new Date(b).getTime() - new Date(a).getTime());
        const bins = tsList.slice(0, limitBins);

        const results = [];

        for (const ts of bins) {
            const rows = await coll.find({ [FIELD_TS]: ts }, { projection: { _id: 0 } }).toArray();

            const byCorr = new Map();

            for (const r of rows) {
                const prob = toNumberOrNull(r[FIELD_PROB]);
                if (prob === null) continue;

                const corridorId = safeCorridorId(r);
                if (!byCorr.has(corridorId)) {
                    byCorr.set(corridorId, {
                        corridor_id: corridorId,
                        timestamp: ts,
                        corridor_risk: 0,
                        lanes_above_thr: 0,
                        lane_count: 0,
                        top_lanes: [],
                    });
                }

                const agg = byCorr.get(corridorId);
                agg.lane_count += 1;
                if (prob > agg.corridor_risk) agg.corridor_risk = prob;
                if (prob >= thr) agg.lanes_above_thr += 1;

                agg.top_lanes.push({
                    SID: r[FIELD_SID],
                    LANE_NAME: r[FIELD_LANE],
                    crash_probability: prob,
                    alert: Boolean(r[FIELD_ALERT]),
                    top_reasons_resolved: buildTopReasons(r),
                });
            }

            const corridors = Array.from(byCorr.values())
                .map((c) => {
                    c.top_lanes.sort((a, b) => b.crash_probability - a.crash_probability);
                    c.top_lanes = c.top_lanes.slice(0, topk);
                    c.is_alert = c.lanes_above_thr >= minLanes && c.corridor_risk >= thr;
                    return c;
                })
                .filter((c) => c.is_alert)
                .sort((a, b) => b.corridor_risk - a.corridor_risk);

            if (corridors.length > 0) results.push({ timestamp: ts, corridors });
        }

        res.json({
            ok: true,
            from,
            to,
            thr,
            topk,
            min_lanes: minLanes,
            bins_considered: bins.length,
            bin_alert_count: results.length,
            results,
        });
    } catch (err) {
        res.status(400).json({ ok: false, error: String(err?.message || err) });
    }
});

/**
 * Latest snapshot for one corridor (SEGMENTS)
 */
app.get("/api/corridors/:corridor_id/latest", mustHaveCollection, async (req, res) => {
    try {
        const corridorId = String(req.params.corridor_id || "").trim();
        if (!corridorId) return res.status(400).json({ ok: false, error: "Missing corridor_id" });

        const q = z
            .object({
                thr: z.string().optional(),
                topk: z.string().optional(),
            })
            .parse(req.query);

        const thr = q.thr ? Number(q.thr) : 0.6;
        const topk = clampInt(parseInt(q.topk || "10", 10), 1, 50);

        const latestTs = await getLatestTimestamp();
        if (!latestTs) return res.json({ ok: true, timestamp: null, corridor: null });

        const rows = await coll
            .find({ [FIELD_TS]: latestTs, [FIELD_SEGMENTS]: corridorId }, { projection: { _id: 0 } })
            .toArray();

        const lanes = [];
        for (const r of rows) {
            const prob = toNumberOrNull(r[FIELD_PROB]);
            if (prob === null) continue;

            lanes.push({
                SID: r[FIELD_SID],
                LANE_NAME: r[FIELD_LANE],
                crash_probability: prob,
                alert: Boolean(r[FIELD_ALERT]),
                is_alert_by_thr: prob >= thr,
                Detector_LATITUDE: r.Detector_LATITUDE,
                Detector_LONGITUDE: r.Detector_LONGITUDE,
                top_reasons_resolved: buildTopReasons(r),
            });
        }

        lanes.sort((a, b) => b.crash_probability - a.crash_probability);
        const topLanes = lanes.slice(0, topk);
        const corridorRisk = topLanes.length ? topLanes[0].crash_probability : 0;

        res.json({
            ok: true,
            timestamp: latestTs,
            corridor: {
                corridor_id: corridorId,
                corridor_risk: corridorRisk,
                lanes_count: lanes.length,
                lanes_above_thr: lanes.filter((x) => x.is_alert_by_thr).length,
                top_lanes: topLanes,
            },
        });
    } catch (err) {
        res.status(400).json({ ok: false, error: String(err?.message || err) });
    }
});

// Start server
(async () => {
    try {
        await connectMongo();
        app.listen(API_PORT, () => {
            console.log(
                `Crash Risk API on ${API_PORT} | Mongo: ${MONGO_URI} | ${MONGO_DB}.${MONGO_COLLECTION} | prob=${FIELD_PROB} alert=${FIELD_ALERT}`
            );
        });
    } catch (err) {
        console.error("Failed to start Crash Risk API:", err);
        process.exit(1);
    }
})();

// Graceful shutdown
process.on("SIGINT", async () => {
    try {
        await client.close();
    } finally {
        process.exit(0);
    }
});
process.on("SIGTERM", async () => {
    try {
        await client.close();
    } finally {
        process.exit(0);
    }
});
