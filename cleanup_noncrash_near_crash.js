/**
 * cleanup_noncrash_near_crash.js
 *
 * Removes non-crash rows that are within ±60 minutes
 * of a crash at the same (LANE_NAME + SID).
 *
 * Node 18+, CommonJS
 * npm i xlsx
 */

const path = require("path");
const xlsx = require("xlsx");

// ================= CONFIG =================
const CRASH_FILE = path.join(
  __dirname,
  "CrashData_WithWeather_2024_2025_FINAL.xlsx",
);

const NONCRASH_FILE = path.join(
  __dirname,
  "NonCrash_Sample10pct_2024_2025.xlsx",
);

const OUTPUT_FILE = path.join(
  __dirname,
  "NonCrash_Sample10pct_2024_2025_CLEAN.xlsx",
);

const TIME_COL = "ROLLUP_TIMESTAMP";
const LANE_COL = "LANE_NAME";
const SID_COL = "SID";

const WINDOW_MINUTES = 60;
// ========================================

// ---------- Helpers ----------
function readRows(xlsxPath) {
  const wb = xlsx.readFile(xlsxPath, { cellDates: true });
  const ws = wb.Sheets[wb.SheetNames[0]];
  return xlsx.utils.sheet_to_json(ws, { defval: null });
}

function toDate(v) {
  if (v instanceof Date && !isNaN(v)) return v;
  if (typeof v === "number") {
    const excelEpoch = new Date(Date.UTC(1899, 11, 30));
    return new Date(excelEpoch.getTime() + v * 86400000);
  }
  if (v === null || v === undefined) return null;
  const d = new Date(String(v).replace(" ", "T"));
  return isNaN(d) ? null : d;
}

// ---------- Main ----------
(function main() {
  console.log("Loading crash data...");
  const crashRows = readRows(CRASH_FILE);

  console.log("Loading non-crash data...");
  const nonCrashRows = readRows(NONCRASH_FILE);

  console.log(`Crash rows: ${crashRows.length}`);
  console.log(`Non-crash rows (before): ${nonCrashRows.length}`);

  // Build crash time index by (LANE|SID)
  const crashIndex = new Map();

  for (const r of crashRows) {
    const lane = String(r[LANE_COL] || "").trim();
    const sid = String(r[SID_COL] || "").trim();
    const t = toDate(r[TIME_COL]);
    if (!lane || !sid || !t) continue;

    const key = `${lane}||${sid}`;
    if (!crashIndex.has(key)) crashIndex.set(key, []);
    crashIndex.get(key).push(t.getTime());
  }

  // Sort crash times for each key
  for (const times of crashIndex.values()) {
    times.sort((a, b) => a - b);
  }

  const WINDOW_MS = WINDOW_MINUTES * 60 * 1000;

  let removed = 0;
  const cleaned = [];

  for (const r of nonCrashRows) {
    const lane = String(r[LANE_COL] || "").trim();
    const sid = String(r[SID_COL] || "").trim();
    const t = toDate(r[TIME_COL]);

    if (!lane || !sid || !t) {
      cleaned.push(r);
      continue;
    }

    const key = `${lane}||${sid}`;
    const crashTimes = crashIndex.get(key);

    if (!crashTimes) {
      cleaned.push(r);
      continue;
    }

    const tt = t.getTime();

    // Binary-search style check
    let contaminated = false;
    for (const ct of crashTimes) {
      if (Math.abs(ct - tt) <= WINDOW_MS) {
        contaminated = true;
        break;
      }
      if (ct > tt + WINDOW_MS) break;
    }

    if (contaminated) {
      removed++;
    } else {
      cleaned.push(r);
    }
  }

  console.log(`Removed contaminated non-crash rows: ${removed}`);
  console.log(`Non-crash rows (after): ${cleaned.length}`);

  // Write output
  const outWb = xlsx.utils.book_new();
  const outWs = xlsx.utils.json_to_sheet(cleaned);
  xlsx.utils.book_append_sheet(outWb, outWs, "noncrash_clean");

  xlsx.writeFile(outWb, OUTPUT_FILE);
  console.log("✅ Wrote cleaned non-crash file:");
  console.log(OUTPUT_FILE);
})();
