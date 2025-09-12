#!/usr/bin/env python3
"""
Activity Cleaning + Visualization Pipeline

Guided by user rules in diaw.txt header message. Non-destructive: writes to out/ only.

Outputs:
- out/cleaned.csv: normalized, annotated records
- out/quality_report.json: validation errors and auto-fix notes
- out/metrics_summary.json: requested metrics
- out/suggestions.txt: typo/normalization suggestions
- out/timesheet_5min.csv: 5-minute grid across all days
- out/weekly_histograms/week_{year}_{week:02d}.png: weekly stacked bars (5-category)

Notes:
- No deletions and no overwrites of existing week_XX_histogram.png at project root; we write under out/.
"""

from __future__ import annotations

import csv
import dataclasses as dc
import json
import os
import re
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import math

try:
    import pandas as pd
except Exception:  # pragma: no cover - soft dependency check
    pd = None  # We'll guard usage and give install hint at runtime

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - soft dependency check
    plt = None


# ---------------------------
# Config and constants
# ---------------------------

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "diaw.txt"
OUT_DIR = ROOT / "out"
HIST_DIR = OUT_DIR / "weekly_histograms"


# Categories per guide
# Rule: Ensure second column contains only F, S, R, P, GOD for validation (Program block).
VALID_5_CATS = {"F", "S", "R", "P", "GOD"}

# We still track the full set observed in the raw (for metrics), incl. W & E and special LO.
FULL_CATEGORIES = VALID_5_CATS | {"W", "E", "LO"}

# Categories for visualization (stacked). Keep GOD visible plus core set.
STACK_CATEGORIES = ["GOD", "P", "R", "E", "S", "F"]


@dc.dataclass
class SegmentHeader:
    # Rule: Segment header must be: MM/DD/YY <DOW> <time-range> (military, no colon)
    raw_date: str
    dow: str
    range_raw: str  # may be invalid; we'll correct per rule
    # Parsed fields
    date_obj: date
    range_start: Optional[int] = None  # HHMM
    range_end: Optional[int] = None  # HHMM
    # Derived
    corrected: bool = False


@dc.dataclass
class Activity:
    # Represents a line like: 2100 P notes
    time_hhmm: int
    category: str  # uppercased
    description: str  # lowercased, normalized
    raw_line: str
    line_no: int

    # Derived/annotations
    errors: List[str] = dc.field(default_factory=list)
    fixes: List[str] = dc.field(default_factory=list)


@dc.dataclass
class DaySegment:
    header: SegmentHeader
    activities: List[Activity] = dc.field(default_factory=list)


# ---------------------------
# Helpers
# ---------------------------

_HEADER_RE = re.compile(
    r"^(?P<mdy>\d{1,2}/\d{1,2}/\d{2})\s+(?P<dow>Su|Sa|Th|Tu|Su|Mo|Tu|We|Th|Fr|Sa|Su|M|T|W|Th|F|Sa|Su|Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(?P<range>[0-9-]+)$",
    re.IGNORECASE,
)
_HEADER_SIMPLE_RE = re.compile(
    r"^(?P<mdy>\d{1,2}/\d{1,2}/\d{2})\s+(?P<dow>M|T|W|Th|F|Sa|Su)\s+(?P<range>[0-9-]+)$",
    re.IGNORECASE,
)

_ACT_RE = re.compile(r"^(?P<time>\d{3,4})\s+(?P<cat>[A-Za-z]+)\s+(?P<desc>.+)$")


def _parse_mdy(mdy: str) -> date:
    # Accept both zero-padded and non-padded
    return datetime.strptime(mdy, "%m/%d/%y").date()


def hhmm_to_minutes(hhmm: int) -> int:
    hh = hhmm // 100
    mm = hhmm % 100
    return hh * 60 + mm


def minutes_to_hhmm(mins: int) -> int:
    mins = mins % (24 * 60)
    hh = mins // 60
    mm = mins % 60
    return hh * 100 + mm


def normalize_description(desc: str) -> Tuple[str, List[str]]:
    """
    Apply substitutions and lower-case per rules.
    Returns (normalized_desc, suggestions)

    Rules implemented:
    - Eliminate all caps in the third column -> lower()
    - Substitutions regardless of capitalization:
      * hmwrk -> hw
      * bathroom -> br
      * mlc -> hw stat 3160
      * treadmill -> tm (and 'tm and' -> 'tm')
      * quiet time -> qt
      * lights out -> lo
    - Minor typo suggestions (heuristic)
    """
    s = desc.strip().lower()

    # Core replacements
    repl = [
        (r"\bhmwrk\b", "hw"),
        (r"\bbathroom\b", "br"),
        (r"\bmlc\b", "hw stat 3160"),
        (r"\btreadmill\b", "tm"),
        (r"\btm and\b", "tm"),
        (r"\bquiet time\b", "qt"),
        (r"\blights out\b", "lo"),
        (r"\bdressup\b", "dress up"),
        (r"\bclass ?time\b", "class"),
        (r"\byoutube\b", "yt"),
    ]
    for pat, rep in repl:
        s = re.sub(pat, rep, s)

    suggestions: List[str] = []
    # Heuristic typo flags
    if "…" in desc or "\u2026" in desc:
        suggestions.append("Replace ellipsis character in description with '...' or remove")
        s = s.replace("…", "...")
    if "dressup" in desc.lower():
        suggestions.append("Use 'dress up' instead of 'dressup'")
    if "comm" in s and "communication" not in s:
        # This is acceptable shorthand; keep as-is but note once
        pass

    return s, suggestions


def parse_file(lines: Iterable[Tuple[int, str]]) -> List[DaySegment]:
    segments: List[DaySegment] = []
    header: Optional[SegmentHeader] = None
    current: Optional[DaySegment] = None

    for line_no, line in lines:
        raw = line.rstrip("\n")
        if not raw.strip():
            continue

        # Header detection
        m = _HEADER_SIMPLE_RE.match(raw)
        if m:
            mdy = m.group("mdy")
            dow = m.group("dow")
            rng = m.group("range")
            date_obj = _parse_mdy(mdy)
            header = SegmentHeader(raw_date=mdy, dow=dow, range_raw=rng, date_obj=date_obj)
            current = DaySegment(header=header)
            segments.append(current)
            continue

        # Activity line
        m = _ACT_RE.match(raw)
        if m and current is not None:
            time_str = m.group("time")
            cat = m.group("cat").upper()
            desc_raw = m.group("desc")
            # Rule: Capitalize all letters in the second column (categories uppercase)
            # Rule: Eliminate all caps in the third column (descriptions lowercase + normalize)
            desc_norm, sugg = normalize_description(desc_raw)
            act = Activity(time_hhmm=int(time_str), category=cat, description=desc_norm, raw_line=raw, line_no=line_no)
            # Attach suggestions to fixes list (informational)
            for s in sugg:
                act.fixes.append(f"suggestion: {s}")
            current.activities.append(act)
            continue

        # Unrecognized line: attach to last segment as error context
        if current is not None:
            # Create a pseudo-activity to capture the anomaly
            act = Activity(time_hhmm=-1, category="?", description=raw, raw_line=raw, line_no=line_no)
            act.errors.append("Unrecognized line format (not header or activity)")
            current.activities.append(act)

    return segments


def validate_and_fix(segments: List[DaySegment]) -> Tuple[List[DaySegment], dict]:
    """
    Apply validation and specific fixes per rules. Returns (segments, quality_report)

    Implemented rules:
    - If there are not exactly 3 strings (time, category, desc) in activity rows, report.
    - Timestamps must be descending; allow a permissive cross at midnight; flag increases after midnight.
    - Ensure second column is one of F, S, R, P, GOD (for validation). Accept LO as special; keep E/W for metrics but include a warning.
    - If header range is not TTTT-TTTT, replace with last -> first timestamps.
    - If desc contains 'notes', ensure category is 'P' and not 'F' (auto-fix from F->P, record fix).
    - Apply substitutions (done in parsing).
    - Add suggestions for possible typos.
    """
    quality = {
        "invalid_rows": [],
        "timestamp_errors": [],
        "category_errors": [],
        "header_corrections": [],
        "auto_fixes": [],
        "notes": [],
    }

    for seg in segments:
        # Header range correction
        rng = seg.header.range_raw
        m = re.match(r"^(\d{3,4})-(\d{3,4})$", rng)
        if m:
            seg.header.range_start = int(m.group(1))
            seg.header.range_end = int(m.group(2))
        else:
            # Rule: replace header range with last timestamp -> first timestamp
            if seg.activities:
                first_ts = seg.activities[-1].time_hhmm if seg.activities[-1].time_hhmm >= 0 else None
                last_ts = seg.activities[0].time_hhmm if seg.activities[0].time_hhmm >= 0 else None
                if first_ts is not None and last_ts is not None:
                    seg.header.range_start = first_ts
                    seg.header.range_end = last_ts
                    seg.header.corrected = True
                    quality["header_corrections"].append(
                        {
                            "date": seg.header.raw_date,
                            "old": rng,
                            "new": f"{first_ts:04d}-{last_ts:04d}",
                            "reason": "Header range not TTTT-TTTT; replaced with last->first timestamps",
                        }
                    )
                else:
                    seg.header.range_start = None
                    seg.header.range_end = None
            else:
                seg.header.range_start = None
                seg.header.range_end = None

        # Validate activities
        prev_time: Optional[int] = None
        after_midnight = True  # We traverse as given (reverse chronological). First entries are typically after midnight.
        crossed_midnight_once = False
        for idx, act in enumerate(seg.activities):
            # Skip malformed pseudo-activities
            if act.time_hhmm < 0:
                quality["invalid_rows"].append(
                    {
                        "date": seg.header.raw_date,
                        "line_no": act.line_no,
                        "raw": act.raw_line,
                        "reason": "Unrecognized activity format",
                    }
                )
                continue

            # Rule: Exactly 3 columns with text (time, category, desc)
            # Our parser ensures at least these; if description empty or cat empty, flag
            if not act.description.strip() or not act.category.strip():
                act.errors.append("Missing category or description")
                quality["invalid_rows"].append(
                    {
                        "date": seg.header.raw_date,
                        "line_no": act.line_no,
                        "raw": act.raw_line,
                        "reason": "Missing category or description",
                    }
                )

            # Rule: Ensure allowed categories; LO is special marker
            if act.category not in FULL_CATEGORIES:
                act.errors.append("Unknown category")
                quality["category_errors"].append(
                    {
                        "date": seg.header.raw_date,
                        "line_no": act.line_no,
                        "raw": act.raw_line,
                        "category": act.category,
                        "message": "Category not in known set {F,S,R,P,GOD,E,W,LO}",
                    }
                )
            # For validation per rule, warn if not in 5 categories (excluding LO) but don't drop
            if act.category not in VALID_5_CATS and act.category not in {"LO"}:
                quality["notes"].append(
                    {
                        "date": seg.header.raw_date,
                        "line_no": act.line_no,
                        "raw": act.raw_line,
                        "message": "Category outside 5-cat validation set; retained for metrics",
                    }
                )

            # Rule: If description mentions 'notes', ensure category is P (auto-fix from F->P only)
            if "notes" in act.description and act.category == "F":
                old = act.category
                act.category = "P"
                act.fixes.append("category: F->P due to 'notes' in description")
                quality["auto_fixes"].append(
                    {
                        "date": seg.header.raw_date,
                        "line_no": act.line_no,
                        "raw": act.raw_line,
                        "fix": "category F->P (notes)",
                    }
                )

            # Rule: Timestamps descending, allow midnight crossover
            if prev_time is not None:
                if act.time_hhmm <= prev_time:
                    # OK: descending or equal (equal still a potential issue?) - Rule says greater than or equal triggers error; equal is not allowed.
                    if act.time_hhmm == prev_time:
                        quality["timestamp_errors"].append(
                            {
                                "date": seg.header.raw_date,
                                "prev": prev_time,
                                "curr": act.time_hhmm,
                                "prev_line": seg.activities[idx - 1].raw_line,
                                "curr_line": act.raw_line,
                                "message": "Timestamps are not chronological (equal)",
                            }
                        )
                        act.errors.append("Equal timestamp to previous")
                else:
                    # Potential midnight crossover if previous was near 0000 and now jumps to 23xx
                    if prev_time <= 200 and act.time_hhmm >= 1200 and not crossed_midnight_once:
                        crossed_midnight_once = True
                        after_midnight = False
                    else:
                        # Rule: Throw error for unchronological sequences after midnight
                        quality["timestamp_errors"].append(
                            {
                                "date": seg.header.raw_date,
                                "prev": prev_time,
                                "curr": act.time_hhmm,
                                "prev_line": seg.activities[idx - 1].raw_line,
                                "curr_line": act.raw_line,
                                "message": "Timestamps are not chronological",
                            }
                        )
                        act.errors.append("Timestamps are not chronological")

            prev_time = act.time_hhmm

    return segments, quality


def iter_lines(path: Path) -> Iterable[Tuple[int, str]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            yield i, line


def compute_durations(seg: DaySegment) -> List[int]:
    """Return per-activity durations in minutes to the next record boundary.

    Uses header range start/end for the boundary activities; LO marks sleep to range end.
    """
    times = [a.time_hhmm for a in seg.activities if a.time_hhmm >= 0]
    if not times:
        return []

    # Convert to an unrolled minute timeline in reverse order (as listed)
    # We'll compute delta to next entry considering midnight wrap.
    deltas: List[int] = []
    for i, t_curr in enumerate(times):
        if i + 1 < len(times):
            t_next = times[i + 1]
            # Since order is typically descending with a single midnight wrap allowed, compute delta accordingly
            if t_next <= t_curr:
                delta = hhmm_to_minutes(t_curr) - hhmm_to_minutes(t_next)
            else:
                # wrap across midnight (e.g., 0000 -> 2345)
                delta = (hhmm_to_minutes(t_curr) + 24 * 60) - hhmm_to_minutes(t_next)
        else:
            # Last activity "earliest" in day; use header range start if available
            if seg.header.range_start is not None:
                t_next = seg.header.range_start
                if t_next <= t_curr:
                    delta = hhmm_to_minutes(t_curr) - hhmm_to_minutes(t_next)
                else:
                    delta = (hhmm_to_minutes(t_curr) + 24 * 60) - hhmm_to_minutes(t_next)
            else:
                # Fallback: zero
                delta = 0
        deltas.append(max(delta, 0))
    return deltas


def to_dataframe(segments: List[DaySegment]):
    if pd is None:
        raise RuntimeError("pandas is required; please install pandas before running the pipeline.")

    rows = []
    for seg in segments:
        deltas = compute_durations(seg)
        delta_idx = 0
        for act in seg.activities:
            if act.time_hhmm < 0:
                continue
            dur_min = deltas[delta_idx] if delta_idx < len(deltas) else 0
            delta_idx += 1
            rows.append(
                {
                    "date": seg.header.date_obj.isoformat(),
                    "dow": seg.header.dow,
                    "range": f"{(seg.header.range_start or 0):04d}-{(seg.header.range_end or 0):04d}",
                    "time": f"{act.time_hhmm:04d}",
                    "category": act.category,
                    "description": act.description,
                    "duration_min": dur_min,
                    "is_lo": int(act.category == "LO"),
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        # Enrich
        df["yyyy_mm_dd"] = pd.to_datetime(df["date"]).dt.date
        df["week"] = pd.to_datetime(df["date"]).dt.isocalendar().week.astype(int)
        df["year"] = pd.to_datetime(df["date"]).dt.isocalendar().year.astype(int)
        df["time_hm"] = df["time"].astype(int)
        df["category5_valid"] = df["category"].isin(list(VALID_5_CATS) + ["LO"])  # LO allowed marker
    return df


def ensure_dirs():
    OUT_DIR.mkdir(exist_ok=True)
    HIST_DIR.mkdir(parents=True, exist_ok=True)


def save_quality_report(quality: dict):
    ensure_dirs()
    (OUT_DIR / "quality_report.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")


def save_cleaned(df):
    ensure_dirs()
    df.to_csv(OUT_DIR / "cleaned.csv", index=False)


def compute_metrics(df):
    """Compute requested metrics as best-effort based on cleaned data."""
    if df.empty:
        return {}

    def sum_minutes(mask):
        return int(df.loc[mask, "duration_min"].sum())

    metrics = {}
    # Category totals (all categories present)
    cat_minutes = df.groupby("category")["duration_min"].sum().sort_values(ascending=False).to_dict()
    metrics["category_totals_min"] = {k: int(v) for k, v in cat_minutes.items()}

    # Research variants (case-insensitive in description)
    metrics["research_minutes"] = sum_minutes(df["description"].str.contains(r"\bresearch\b", case=False, na=False))

    # 'tm' treadmill minutes overall
    metrics["tm_minutes"] = sum_minutes(df["description"].str.contains(r"\btm\b", na=False))

    # Homework ITSC vs general ITSC
    metrics["hw_itsc_minutes"] = sum_minutes(df["description"].str.contains(r"\bhw\s+itsc\b", na=False))
    metrics["itsc_minutes"] = sum_minutes(
        df["description"].str.contains(r"\bitsc\b", na=False)
        & ~df["description"].str.contains(r"\bhw\s+itsc\b", na=False)
    )

    # STAT vs hw STAT
    metrics["hw_stat_minutes"] = sum_minutes(df["description"].str.contains(r"\bhw\s+stat\b", na=False))
    metrics["stat_minutes"] = sum_minutes(
        df["description"].str.contains(r"\bstat\b", na=False)
        & ~df["description"].str.contains(r"\bhw\s+stat\b", na=False)
    )

    # Dress up
    metrics["dress_up_minutes"] = sum_minutes(df["description"].str.contains(r"\bdress\s*up\b", na=False))

    # Poop
    metrics["poop_minutes"] = sum_minutes(df["description"].str.contains(r"\bpoop\b", na=False))

    # GOD category minutes (totals, weekly, avg per day)
    god_df = df[df["category"] == "GOD"]
    metrics["god_total_minutes"] = int(god_df["duration_min"].sum())
    if not god_df.empty:
        god_week = god_df.groupby(["year", "week"])['duration_min'].sum().astype(int).to_dict()
        metrics["god_minutes_per_week"] = {f"{y}-W{w:02d}": m for (y, w), m in god_week.items()}
        god_day = god_df.groupby(["date"])['duration_min'].sum().astype(int)
        metrics["god_avg_minutes_per_day"] = float(god_day.mean()) if len(god_day) else 0.0
        # 'pray' counts within GOD (exact occurrences)
        pray_mask = god_df["description"].str.contains(r"\bpray\b", na=False)
        metrics["god_pray_occurrences"] = int(pray_mask.sum())
        metrics["god_pray_minutes"] = int(god_df.loc[pray_mask, "duration_min"].sum())
    else:
        metrics["god_minutes_per_week"] = {}
        metrics["god_avg_minutes_per_day"] = 0.0
        metrics["god_pray_occurrences"] = 0
        metrics["god_pray_minutes"] = 0

    # Totals per description (cleaned)
    desc_minutes = (
        df.groupby("description")["duration_min"].sum().sort_values(ascending=False).to_dict()
    )
    metrics["description_totals_min"] = {k: int(v) for k, v in desc_minutes.items()}

    # Classtime heuristic: 4 letters + 4 digits pattern
    class_code_re = re.compile(r"\b[A-Za-z]{4}\s*\d{4}\b")
    code_mask = df["description"].str.contains(class_code_re, na=False)
    metrics["classcode_minutes"] = sum_minutes(code_mask)
    # Study/treadmill minutes for lines with class codes but with extra descriptors (like 'hw' or 'tm')
    extra_mask = code_mask & df["description"].str.contains(r"\bhw\b|\btm\b|\bnotes\b", na=False)
    metrics["classcode_study_or_tm_minutes"] = sum_minutes(extra_mask)

    # Awake vs asleep per week using LO and header range end
    sleep_per_day = {}
    for (d, g) in df.groupby("date"):
        day_df = g.sort_values("time_hm")
        # LO entries are special; take the first one after midnight (<= 0100 typically) if present
        lo_rows = day_df[day_df["category"] == "LO"]
        if not lo_rows.empty:
            lo_time = int(lo_rows.iloc[0]["time"])  # HHMM
            # End of day from header range end
            rng = day_df.iloc[0]["range"]
            m = re.match(r"(\d{4})-(\d{4})", str(rng))
            if m:
                range_end = int(m.group(2))
                # Minutes from LO -> end (account for wrap if needed)
                if range_end >= lo_time:
                    sleep_min = hhmm_to_minutes(range_end) - hhmm_to_minutes(lo_time)
                else:
                    sleep_min = 24 * 60 - (hhmm_to_minutes(lo_time) - hhmm_to_minutes(range_end))
            else:
                sleep_min = 0
        else:
            sleep_min = 0
        awake_min = int(day_df[day_df["category"] != "LO"]["duration_min"].sum())
        sleep_per_day[d] = {"sleep_min": int(max(sleep_min, 0)), "awake_min": int(awake_min)}
    # Aggregate weekly
    awake_week = defaultdict(int)
    sleep_week = defaultdict(int)
    for d_str, mins in sleep_per_day.items():
        dt = datetime.strptime(d_str, "%Y-%m-%d").date()
        iso = dt.isocalendar()
        key = f"{iso.year}-W{iso.week:02d}"
        awake_week[key] += mins["awake_min"]
        sleep_week[key] += mins["sleep_min"]
    metrics["awake_minutes_per_week"] = dict(awake_week)
    metrics["sleep_minutes_per_week"] = dict(sleep_week)

    # Comm vs other social vs ignite
    social = df[df["category"] == "S"]
    metrics["social_comm_minutes"] = int(social[social["description"].str.contains(r"\bcomm\b", na=False)]["duration_min"].sum())
    metrics["social_ignite_minutes"] = int(social[social["description"].str.contains(r"ignite", na=False)]["duration_min"].sum())
    metrics["social_other_minutes"] = int(social[~social["description"].str.contains(r"\bcomm\b|ignite", na=False)]["duration_min"].sum())

    # Workout minutes (W category)
    metrics["workout_minutes"] = int(df[df["category"] == "W"]["duration_min"].sum())

    # Volunteer minutes (keywords)
    metrics["volunteer_minutes"] = sum_minutes(df["description"].str.contains(r"\bvolunteer\b", na=False))

    return metrics


def save_metrics(metrics: dict):
    ensure_dirs()
    (OUT_DIR / "metrics_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def save_suggestions(segments: List[DaySegment]):
    ensure_dirs()
    suggestions: List[str] = []
    for seg in segments:
        for act in seg.activities:
            for fix in act.fixes:
                if fix.startswith("suggestion:"):
                    suggestions.append(f"{seg.header.raw_date} {act.raw_line} -> {fix}")
    if suggestions:
        (OUT_DIR / "suggestions.txt").write_text("\n".join(suggestions), encoding="utf-8")
    else:
        (OUT_DIR / "suggestions.txt").write_text("No suggestions.", encoding="utf-8")


def plot_weekly_histograms(df):
    """Produce stacked bar charts over sequential 7-day blocks derived from data order.

    User revert request:
    - Return to stacked bars.
    - Use the original day headers order (not ISO weeks).
    - Partition the timeline into consecutive 7-day stints starting from the first date present.
    - Each chart must show exactly 7 days as listed (even if gaps -> zero bars).
    - Day labels use the recorded day-of-week tokens from headers (fallback to date weekday if missing).
    """
    if df.empty or plt is None:
        return []

    saved = []
    # Sort by actual chronological date
    df_sorted = df.sort_values("yyyy_mm_dd")
    unique_dates = list(dict.fromkeys(df_sorted["yyyy_mm_dd"].tolist()))  # preserve order

    # Create mapping from date -> header dow (first encountered)
    date_to_dow = {}
    for row in df_sorted[['yyyy_mm_dd', 'dow']].drop_duplicates().itertuples(index=False):
        if row[0] not in date_to_dow:
            date_to_dow[row[0]] = row[1]

    colors = {
        "GOD": "#ffd700",
        "P": "#1f77b4",
        "R": "#ff7f0e",
        "E": "#2ca02c",
        "S": "#9467bd",
        "F": "#8c564b",
    }

    # Build 7-day chunks
    for block_idx in range(0, len(unique_dates), 7):
        block_dates = unique_dates[block_idx:block_idx + 7]
        if len(block_dates) < 7:
            # If last partial block should be ignored to maintain 7-day requirement, break
            break

        block_df = df_sorted[df_sorted['yyyy_mm_dd'].isin(block_dates)]
        daily_cat = (
            block_df.groupby(['yyyy_mm_dd', 'category'])['duration_min'].sum().reset_index()
        )
        # Build pivot with all categories
        pivot = daily_cat.pivot_table(index='yyyy_mm_dd', columns='category', values='duration_min', aggfunc='sum').fillna(0)
        for c in STACK_CATEGORIES:
            if c not in pivot.columns:
                pivot[c] = 0
        pivot = pivot[STACK_CATEGORIES]
        # Ensure all seven dates rows present
        for d in block_dates:
            if d not in pivot.index:
                pivot.loc[d] = 0
        pivot = pivot.sort_index()

        # Convert minutes to hours (float)
        pivot_hours = pivot / 60.0

        fig, ax = plt.subplots(figsize=(11, 4.5))
        bottom = [0.0] * len(pivot_hours.index)
        x_vals = list(range(len(pivot_hours.index)))
        for cat in STACK_CATEGORIES:
            vals = pivot_hours[cat].values
            ax.bar(x_vals, vals, bottom=bottom, label=cat, color=colors.get(cat))
            bottom = [b + v for b, v in zip(bottom, vals)]

        # X tick labels: use header dow plus day number for clarity
        labels = []
        for d in pivot_hours.index:
            dow_token = date_to_dow.get(d)
            if dow_token is None:
                dow_token = date.fromisoformat(str(d)).strftime('%a')
            labels.append(f"{dow_token}\n{d}")
        ax.set_xticks(x_vals)
        ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=9)
        ax.set_ylabel('Hours')
        start_date = pivot_hours.index[0]
        end_date = pivot_hours.index[-1]
        ax.set_title(f"Days {start_date} to {end_date} — Stacked Daily Hours")
        ax.legend(ncol=min(len(STACK_CATEGORIES), 6), fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.18))
        ax.grid(axis='y', linestyle=':', alpha=0.4)
        plt.tight_layout()
        ensure_dirs()
        out_path = HIST_DIR / f"block_{block_idx//7 + 1:02d}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        saved.append(str(out_path))
    return saved


def write_timesheet(df):
    """Make a huge CSV with 5-minute slots across all days: date,time_slot,category,description.

    The slot is assigned using the activity's duration expansion. LO slots are marked as 'LO'.
    """
    if df.empty:
        return None
    # Build expansion
    rows = []
    for (d, t, cat, desc, dur) in df[["date", "time", "category", "description", "duration_min"]].itertuples(index=False):
        if dur <= 0:
            continue
        start_min = hhmm_to_minutes(int(t))
        # Expand backwards since listing is reverse-chron; slots go from start_min-1 downwards.
        # We'll allocate 'dur' minutes ending at the next timestamp boundary.
        for m in range(dur):
            mins = (start_min - m - 1) % (24 * 60)
            # Align to 5-minute slots by floor
            slot_start = mins - (mins % 5)
            rows.append((d, f"{minutes_to_hhmm(slot_start):04d}", cat, desc))
    if not rows:
        return None
    ts = pd.DataFrame(rows, columns=["date", "slot", "category", "description"]).drop_duplicates()
    ts = ts.sort_values(["date", "slot"])  # ascending per day
    ts.to_csv(OUT_DIR / "timesheet_5min.csv", index=False)
    return str(OUT_DIR / "timesheet_5min.csv")


def write_weekly_category_totals(df):
    """Save per-week totals per category to CSV."""
    if df.empty:
        return None
    g = (
        df.groupby(["year", "week", "category"])['duration_min']
        .sum()
        .reset_index()
        .sort_values(["year", "week", "category"])
    )
    out_path = OUT_DIR / "weekly_category_totals.csv"
    g.to_csv(out_path, index=False)
    return str(out_path)


def main():
    ensure_dirs()
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")

    segments = parse_file(iter_lines(DATA_FILE))
    segments, quality = validate_and_fix(segments)
    save_quality_report(quality)

    if pd is None:
        print("pandas not installed; install pandas and matplotlib to complete pipeline.")
        return 1

    df = to_dataframe(segments)
    if not df.empty:
        save_cleaned(df)
        metrics = compute_metrics(df)
        save_metrics(metrics)
        save_suggestions(segments)
        write_timesheet(df)
        write_weekly_category_totals(df)
        saved = plot_weekly_histograms(df)
        print(f"Saved {len(saved)} weekly histogram(s) under {HIST_DIR}")
    else:
        print("No activities parsed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
