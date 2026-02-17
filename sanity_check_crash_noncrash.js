/**
 * sanity_check_crash_noncrash.js
 * Node 18+, CommonJS
 *
 * Requires: npm i xlsx
 */

const path = require("path");
const xlsx = require("xlsx");

// ---------------- CONFIG ----------------
const CRASH_FILE = path.join(
  __dirname,
  "CrashData_WithWeather_2024_2025_FINAL.xlsx",
); // change if needed
const NONCRASH_FILE = path.join(
  __dirname,
  "NonCrash_Sample10pct_2024_2025.xlsx",
); // change if needed

const TIME_COL = "ROLLUP_TIMESTAMP";
const KEY_COLS = ["LANE_NAME", "SID", TIME_COL];

const SPEED_COL = "AVERAGE_SPEED";
const VOL_COL = "TOTAL_VOLUME";

// ----------------------------------------
function readFirstSheetRows(fp) {
  const wb = xlsx.readFile(fp, { cellDates: true });
  const ws = wb.Sheets[wb.SheetNames[0]];
  return xlsx.utils.sheet_to_json(ws, { defval: null });
}

function toDate(v) {
  if (!v) return null;
  if (v instanceof Date && !isNaN(v)) return v;
  const d = new Date(String(v));
  return isNaN(d) ? null : d;
}

function ymd(d) {
  if (!d) return null;
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

function minutesDiff(a, b) {
  return Math.round((b.getTime() - a.getTime()) / 60000);
}

function makeKey(r) {
  const lane = String(r.LANE_NAME ?? "").trim();
  const sid = String(r.SID ?? "").trim();
  const dt = toDate(r[TIME_COL]);
  const ts = dt ? dt.toISOString().slice(0, 19) : "";
  return `${lane}||${sid}||${ts}`;
}

function basicReport(name, rows) {
  console.log(`\n==============================`);
  console.log(`DATASET: ${name}`);
  console.log(`Rows: ${rows.length.toLocaleString()}`);

  // Time parse
  let parsed = 0,
    missing = 0;
  const years = new Map();
  const times = [];

  for (const r of rows) {
    const d = toDate(r[TIME_COL]);
    if (!d) {
      missing++;
      continue;
    }
    parsed++;
    years.set(d.getFullYear(), (years.get(d.getFullYear()) || 0) + 1);
    times.push(d);
  }

  times.sort((a, b) => a - b);

  console.log(
    `Parsed timestamps: ${parsed.toLocaleString()} | Missing/bad: ${missing.toLocaleString()}`,
  );
  console.log(
    `Year counts:`,
    Object.fromEntries([...years.entries()].sort((a, b) => a[0] - b[0])),
  );

  if (times.length) {
    console.log(`Min time: ${times[0].toISOString()}`);
    console.log(`Max time: ${times[times.length - 1].toISOString()}`);

    // 15-min step check (sample 5000 diffs max)
    const diffs = [];
    const stepN = Math.min(times.length - 1, 5000);
    for (let i = 1; i <= stepN; i++) {
      diffs.push(minutesDiff(times[i - 1], times[i]));
    }
    const counts = {};
    for (const d of diffs) counts[d] = (counts[d] || 0) + 1;
    const top = Object.entries(counts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 8);
    console.log(
      `Top time gaps (minutes):`,
      top.map(([k, v]) => `${k}:${v}`).join("  "),
    );
  }

  // Column presence
  const cols = new Set(Object.keys(rows[0] || {}));
  console.log(
    `Has key cols:`,
    KEY_COLS.map((c) => `${c}:${cols.has(c) ? "OK" : "MISSING"}`).join("  "),
  );
  console.log(
    `Has speed/vol: ${SPEED_COL}:${cols.has(SPEED_COL) ? "OK" : "MISSING"}  ${VOL_COL}:${cols.has(VOL_COL) ? "OK" : "MISSING"}`,
  );

  return { years };
}

function dupReport(name, rows) {
  const seen = new Set();
  let dups = 0;
  for (const r of rows) {
    const k = makeKey(r);
    if (!k.endsWith("||")) {
      // ignore empty-ts keys
      if (seen.has(k)) dups++;
      else seen.add(k);
    }
  }
  console.log(
    `Duplicates by (LANE_NAME,SID,ROLLUP_TIMESTAMP) in ${name}: ${dups.toLocaleString()}`,
  );
}

function contaminationCheck(crashRows, nonCrashRows, windowMin = 60) {
  // Build non-crash index by lane||sid with sorted times
  const idx = new Map();

  for (const r of nonCrashRows) {
    const lane = String(r.LANE_NAME ?? "").trim();
    const sid = String(r.SID ?? "").trim();
    const d = toDate(r[TIME_COL]);
    if (!lane || !sid || !d) continue;
    const key = `${lane}||${sid}`;
    if (!idx.has(key)) idx.set(key, []);
    idx.get(key).push(d.getTime());
  }

  for (const [k, arr] of idx) arr.sort((a, b) => a - b);

  function hasWithin(arr, t, w) {
    // binary search for closest
    let lo = 0,
      hi = arr.length - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (arr[mid] < t) lo = mid + 1;
      else hi = mid - 1;
    }
    // lo is insertion point
    const candidates = [];
    if (lo < arr.length) candidates.push(arr[lo]);
    if (lo - 1 >= 0) candidates.push(arr[lo - 1]);
    return candidates.some((x) => Math.abs(x - t) <= w * 60000);
  }

  let checked = 0,
    contaminated = 0;
  for (const r of crashRows) {
    const lane = String(r.LANE_NAME ?? "").trim();
    const sid = String(r.SID ?? "").trim();
    const d = toDate(r[TIME_COL]);
    if (!lane || !sid || !d) continue;
    checked++;
    const key = `${lane}||${sid}`;
    const arr = idx.get(key);
    if (!arr || !arr.length) continue;
    if (hasWithin(arr, d.getTime(), windowMin)) contaminated++;
  }

  console.log(
    `Crash contamination check (non-crash within ±${windowMin} min, same lane+sid):`,
  );
  console.log(`  crash rows checked: ${checked.toLocaleString()}`);
  console.log(
    `  contaminated hits:  ${contaminated.toLocaleString()}  (${checked ? ((100 * contaminated) / checked).toFixed(1) : "0"}%)`,
  );
}

// ---------------- RUN ----------------
(function main() {
  console.log("Loading files...");
  const crash = readFirstSheetRows(CRASH_FILE);
  const noncrash = readFirstSheetRows(NONCRASH_FILE);

  const crashInfo = basicReport("CRASH", crash);
  const nonInfo = basicReport("NONCRASH (sample)", noncrash);

  // Check 2023/2022 presence
  const badYearsCrash = Object.keys(crashInfo.years).filter(
    (y) => !["2024", "2025"].includes(y),
  );
  const badYearsNon = Object.keys(nonInfo.years).filter(
    (y) => !["2024", "2025"].includes(y),
  );
  console.log(`\nYear scope check:`);
  console.log(
    `  Crash extra years: ${badYearsCrash.length ? badYearsCrash.join(", ") : "NONE"}`,
  );
  console.log(
    `  Non-crash extra years: ${badYearsNon.length ? badYearsNon.join(", ") : "NONE"}`,
  );

  dupReport("CRASH", crash);
  dupReport("NONCRASH", noncrash);

  // Contamination
  contaminationCheck(crash, noncrash, 60);

  console.log("\n✅ Sanity check complete.");
})();
