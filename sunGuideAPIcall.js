/**
 * SunGuide Rollup API
 * ----------------------------------------------------
 * - Reads 15-min rollup data from SGDB
 * - Uses query from ./queries/sunguideRollupQuery.js
 * - Env vars follow SUNGUIDE_SQL_* convention
 */

require("dotenv").config();

const express = require("express");
const cors = require("cors");
const helmet = require("helmet");
const compression = require("compression");
const morgan = require("morgan");
const sql = require("mssql");

const { buildRollupQuerySql } = require("./queries/sunguideRollupQuery");

const app = express();

// -------------------- Middleware --------------------
app.use(helmet());
app.use(cors());
app.use(express.json({ limit: "2mb" }));
app.use(compression());
app.use(morgan("combined"));

const PORT = Number(process.env.PORT || 8089);

// -------------------- Env Validation --------------------
function requireEnv() {
  const missing = [];
  if (!process.env.SUNGUIDE_SQL_USER) missing.push("SUNGUIDE_SQL_USER");
  if (!process.env.SUNGUIDE_SQL_PASSWORD) missing.push("SUNGUIDE_SQL_PASSWORD");
  if (!process.env.SUNGUIDE_SQL_SERVER) missing.push("SUNGUIDE_SQL_SERVER");
  if (!process.env.SUNGUIDE_SQL_DATABASE) missing.push("SUNGUIDE_SQL_DATABASE");

  if (missing.length) {
    console.error("❌ Missing required env vars:", missing.join(", "));
    process.exit(1);
  }
}

// -------------------- SQL Config --------------------
const sqlConfig = {
  user: process.env.SUNGUIDE_SQL_USER,
  password: process.env.SUNGUIDE_SQL_PASSWORD,
  database: process.env.SUNGUIDE_SQL_DATABASE,
  server: process.env.SUNGUIDE_SQL_SERVER,

  connectionTimeout: Number(process.env.SUNGUIDE_SQL_CONN_TIMEOUT || 60000),
  requestTimeout: Number(process.env.SUNGUIDE_SQL_REQ_TIMEOUT || 300000),

  pool: {
    max: Number(process.env.SUNGUIDE_SQL_POOL_MAX || 15),
    min: Number(process.env.SUNGUIDE_SQL_POOL_MIN || 0),
    idleTimeoutMillis: Number(process.env.SUNGUIDE_SQL_IDLE_TIMEOUT || 30000),
  },

  options: {
    encrypt: String(process.env.SUNGUIDE_SQL_ENCRYPT || "false") === "true",
    trustServerCertificate:
      String(process.env.SUNGUIDE_SQL_TRUST_CERT || "true") === "true",
    trustedConnection: true,
  },
};

// -------------------- SQL Pool (singleton) --------------------
let poolPromise = null;

async function getPool() {
  if (!poolPromise) {
    poolPromise = sql.connect(sqlConfig);
  }
  return poolPromise;
}

// -------------------- Time helpers --------------------
function toDateOrNull(v) {
  if (!v) return null;
  const d = new Date(v);
  return isNaN(d.getTime()) ? null : d;
}

function floorTo15MinUTC(d) {
  const fifteen = 15 * 60 * 1000;
  return new Date(Math.floor(d.getTime() / fifteen) * fifteen);
}

function shiftMinutes(d, mins) {
  return new Date(d.getTime() + mins * 60 * 1000);
}

// -------------------- Routes --------------------
app.get("/", (_req, res) => {
  res.json({
    ok: true,
    service: "sunguide-api",
    endpoints: [
      "/api/sunguide/rollup?start=ISO&end=ISO",
      "/api/sunguide/latest?bins=2",
      "/api/health",
    ],
  });
});

/**
 * GET /api/sunguide/rollup?start=ISO&end=ISO
 */
app.get("/api/sunguide/rollup", async (req, res) => {
  try {
    const start = toDateOrNull(req.query.start);
    const end = toDateOrNull(req.query.end);

    if (!start || !end || end <= start) {
      return res.status(400).json({
        ok: false,
        error: "Use ?start=ISO&end=ISO (end must be > start)",
      });
    }

    const pool = await getPool();
    const query = buildRollupQuerySql();

    const r = await pool
      .request()
      .input("StartDate", sql.DateTime2, start)
      .input("EndDate", sql.DateTime2, end)
      .query(query);

    res.json({
      ok: true,
      start,
      end,
      rows: r.recordset.length,
      data: r.recordset,
    });
  } catch (err) {
    console.error("❌ rollup error:", err);
    res.status(500).json({ ok: false, error: err.message });
  }
});

/**
 * GET /api/sunguide/latest?bins=2
 * Returns last CLOSED 15-min bins
 */
app.get("/api/sunguide/latest", async (req, res) => {
  try {
    const bins = Math.min(Math.max(Number(req.query.bins || 2), 1), 96);

    const now = new Date();
    const end = shiftMinutes(floorTo15MinUTC(now), -15);
    const start = shiftMinutes(end, -15 * bins);

    const pool = await getPool();
    const query = buildRollupQuerySql();

    const r = await pool
      .request()
      .input("StartDate", sql.DateTime2, start)
      .input("EndDate", sql.DateTime2, end)
      .query(query);

    res.json({
      ok: true,
      bins,
      start,
      end,
      rows: r.recordset.length,
      data: r.recordset,
    });
  } catch (err) {
    console.error("❌ latest error:", err);
    res.status(500).json({ ok: false, error: err.message });
  }
});

/**
 * Health check
 */
app.get("/api/health", async (_req, res) => {
  try {
    const pool = await getPool();
    await pool.request().query("SELECT 1");
    res.json({ ok: true, db: true });
  } catch (err) {
    res.status(500).json({ ok: false, db: false });
  }
});

// -------------------- Start server --------------------
(async function start() {
  requireEnv();

  try {
    await getPool();
    console.log("✅ Connected to SGDB");
  } catch (err) {
    console.error("❌ DB connection failed:", err);
    process.exit(1);
  }

  app.listen(PORT, () => {
    console.log(`✅ SunGuide API running on http://localhost:${PORT}`);
  });
})();
