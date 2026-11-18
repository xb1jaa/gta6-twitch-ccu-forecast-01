"""Forecast GTA VI's launch-day Twitch viewers hour by hour.

Method (fitted at run time from the case file alone; no network, no
hard-coded answers, standard library only):

Every past launch in the case's `comparables` gives one release-day
curve.  For each launch we take the log ratio of each hour's viewers
to its hour-1 print, L_h = ln(V_h) - ln(V_1).  Two regularities in
those curves drive the forecast:

1.  The curve's height falls as the hour-1 print grows.  Across the
    launches, the evening log ratio correlates about -0.93 with
    ln(V_1): a launch whose first hour is already huge (the world
    mostly unlocked, or a simultaneous global unlock) stays flat,
    while one whose first hour is a partial sample (rolling midnight
    unlocks, the Americas still to come) climbs steeply into the
    evening.  A per-hour regression of L_h on ln(V_1) captures this.

2.  Rolling-midnight console launches -- the kind GTA VI is -- behave
    differently from simultaneous-global and early-access launches.
    The forecast therefore blends two median models: a rolling-only
    regression (the launches whose unlock pattern matches this
    contest) and a regression on all launches with a rolling
    indicator.  The blend hedges the small-sample noise of the
    rolling-only fit.

The spread is honest, not assumed: the quantiles come from the
empirical residuals of those regressions.  The rolling fit's spread
is local -- the residuals of the launches nearest in ln(V_1) -- so a
huge hour-1 print gets the tight spread big launches show, and a
small one the wide spread small launches show.  Where the target's
hour-1 print lies outside the range the comparables cover, the
median is held at the edge of that range (no linear extrapolation
into unseen territory) and the band is widened in proportion to the
distance outside it.

If the case carries too few usable comparables to fit anything, the
program falls back to a flat persistence forecast so the output is
still a valid forecast file.

Output: {"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}, ...]}
for h = 2..24, written to <id>.out for each <id>.in found.  The run is
deterministic: same input bytes give the same output bytes.
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard-normal quantiles at those levels, used only to widen the
# band symmetrically when extrapolating beyond the observed range.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# Blend weights of the two median models (rolling-only, all-launch
# indicator).  Chosen on the four practice launches and on
# leave-one-out scoring over the rolling comparables.
BLEND_ROLLING = 0.7
BLEND_INDICATOR = 0.3
# How many nearest launches (in ln V_1) supply the local spread.
LOCAL_K = 10
# Band widening per log unit the target's ln(V_1) sits outside the
# comparables' observed range (capped).
WIDEN_PER_UNIT = 0.30
WIDEN_CAP = 0.9
# Fallback persistence spread (the house baseline's figure), used only
# when the case has too few comparables to fit the model at all.
FALLBACK_SIGMA = 0.15


def parse_ts(text):
    """An ISO-8601 UTC timestamp as a datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def round2(value):
    """value rounded to 2 decimals, half up; whole numbers as ints."""
    scaled = value * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    rounded = whole / 100
    return int(rounded) if rounded.is_integer() else rounded


def release_day_viewers(comp):
    """{hour: viewers} for the 24 UTC hours of a comparable's release day.

    Hours before the release day (negative hour numbers) are the
    pre-release baseline and are not returned here.  A comparable
    with no usable release day returns an empty mapping.
    """
    try:
        day_start = parse_ts(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return {}
    out = {}
    for row in comp.get("hourly") or []:
        viewers = row.get("viewers")
        if viewers is None or not math.isfinite(viewers):
            continue
        try:
            when = parse_ts(row["from"])
        except (KeyError, TypeError, ValueError):
            continue
        hour = int((when - day_start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24:
            out[hour] = viewers
    return out


def is_rolling(comp):
    """True when the launch unlocked at midnight local, rolling by timezone."""
    unlock = (comp.get("unlock") or "").lower()
    return "rolling" in unlock or "midnight local" in unlock


def launch_weight(comp):
    """Down-weight launches whose unlock pattern differs from this contest's.

    Early-access launches peaked before their release day, simultaneous
    global unlocks put the whole world live in hour 1, and PC launches
    mix a second unlock clock into the curve.  Each factor is a
    judgement from the unlock notes, applied the same way to every
    launch, never by name.
    """
    unlock = (comp.get("unlock") or "").lower()
    platforms = (comp.get("platforms") or "").lower()
    weight = 1.0
    if "early-access-contaminated" in unlock or "early access" in unlock:
        weight *= 0.35
    if "simultaneous" in unlock or "single global unlock" in unlock:
        weight *= 0.45
    if "pc" in platforms:
        weight *= 0.8
    return weight


def launch_curves(case):
    """One record per usable comparable: hour-1 print and log-ratio curve."""
    curves = []
    for comp in case.get("comparables") or []:
        series = release_day_viewers(comp)
        first = series.get(1)
        if not first or first <= 0:
            continue
        ratios = {}
        for hour in range(2, 25):
            viewers = series.get(hour)
            if viewers is not None and viewers > 0:
                ratios[hour] = math.log(viewers) - math.log(first)
        if len(ratios) < 20:
            continue
        curves.append({
            "log_v1": math.log(first),
            "v1": first,
            "rolling": is_rolling(comp),
            "weight": launch_weight(comp),
            "ratios": ratios,
        })
    return curves


def weighted_ols(xs, ys, ws):
    """Weighted least squares for y ~ a + b*x; returns (a, b)."""
    total = sum(ws)
    if total <= 0 or len(xs) < 2:
        return (sum(ys) / len(ys) if ys else 0.0), 0.0
    mean_x = sum(w * x for x, w in zip(xs, ws)) / total
    mean_y = sum(w * y for y, w in zip(ys, ws)) / total
    sxx = sum(w * (x - mean_x) ** 2 for x, w in zip(xs, ws))
    sxy = sum(w * (x - mean_x) * (y - mean_y)
              for x, y, w in zip(xs, ys, ws))
    slope = sxy / sxx if sxx > 0 else 0.0
    return mean_y - slope * mean_x, slope


def solve3(matrix, rhs):
    """Solve a 3x3 linear system by Gaussian elimination with pivoting."""
    rows = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(rows[r][col]))
        rows[col], rows[pivot] = rows[pivot], rows[col]
        if abs(rows[col][col]) < 1e-12:
            rows[col][col] = 1e-12
        for r in range(col + 1, 3):
            factor = rows[r][col] / rows[col][col]
            for cc in range(col, 4):
                rows[r][cc] -= factor * rows[col][cc]
    beta = [0.0, 0.0, 0.0]
    for i in (2, 1, 0):
        beta[i] = (rows[i][3] -
                   sum(rows[i][j] * beta[j] for j in range(i + 1, 3))) / rows[i][i]
    return beta


def fit_indicator(rows):
    """Weighted least squares for y ~ b0 + b1*x1 + b2*x2.

    rows are (x1, x2, y, weight).  Returns (beta, sorted residuals).
    """
    design = [[1.0, r[0], r[1]] for r in rows]
    ys = [r[2] for r in rows]
    ws = [r[3] for r in rows]
    normal = [[sum(w * design[k][i] * design[k][j]
                   for k, w in enumerate(ws)) for j in range(3)]
              for i in range(3)]
    rhs = [sum(w * design[k][i] * ys[k]
               for k, w in enumerate(ws)) for i in range(3)]
    beta = solve3(normal, rhs)
    residuals = sorted(
        y - sum(beta[j] * row[j] for j in range(3))
        for row, y in zip(design, ys))
    return beta, residuals


def quantile(sorted_values, tau):
    """Linearly interpolated tau-quantile of an ascending list."""
    count = len(sorted_values)
    if count == 0:
        return 0.0
    if count == 1:
        return sorted_values[0]
    pos = tau * (count - 1)
    low = int(pos)
    high = min(count - 1, low + 1)
    frac = pos - low
    return sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac


def clip(value, low, high):
    return min(high, max(low, value))


def median_log_ratios(curves, log_v1):
    """Per-hour median log ratios: the blend of the two regressions.

    Returns ({hour: [5 log-ratio quantiles]}, hi_log_v1, lo_log_v1)
    where the log-v1 bounds are the comparables' observed range, for
    the caller's extrapolation widening.
    """
    rolling = [c for c in curves if c["rolling"]]
    use_rolling = rolling if len(rolling) >= 5 else curves
    all_log = [c["log_v1"] for c in curves]
    lo_all, hi_all = min(all_log), max(all_log)
    log_v1 = clip(log_v1, lo_all, hi_all)

    medians = {}
    for hour in range(2, 25):
        # Component 1: rolling-only regression, local spread.
        xs, ys, ws = [], [], []
        for curve in use_rolling:
            if hour in curve["ratios"]:
                xs.append(curve["log_v1"])
                ys.append(curve["ratios"][hour])
                ws.append(curve["weight"])
        if len(xs) >= 5:
            intercept, slope = weighted_ols(xs, ys, ws)
            roll_lo = min(xs)
            roll_hi = max(xs)
            median_roll = intercept + slope * clip(log_v1, roll_lo, roll_hi)
            residuals = sorted(
                (y - (intercept + slope * x), x)
                for x, y in zip(xs, ys))
            nearest = sorted(residuals, key=lambda r: abs(r[1] - log_v1))
            nearest = nearest[:LOCAL_K]
            spread = sorted(r[0] for r in nearest)
            comp1 = [median_roll + quantile(spread, tau) for tau in LEVELS]
        else:
            comp1 = None

        # Component 2: all launches with a rolling indicator; the
        # target is a rolling launch, so its indicator is 1.
        rows = [(c["log_v1"], 1.0 if c["rolling"] else 0.0,
                 c["ratios"][hour], c["weight"])
                for c in curves if hour in c["ratios"]]
        if len(rows) >= 5:
            beta, residuals = fit_indicator(rows)
            median_ind = (beta[0] + beta[1] * log_v1 + beta[2] * 1.0)
            comp2 = [median_ind + quantile(residuals, tau) for tau in LEVELS]
        else:
            comp2 = None

        if comp1 is not None and comp2 is not None:
            medians[hour] = [BLEND_ROLLING * a + BLEND_INDICATOR * b
                             for a, b in zip(comp1, comp2)]
        elif comp1 is not None:
            medians[hour] = comp1
        elif comp2 is not None:
            medians[hour] = comp2
        else:
            # No fit for this hour: hold the previous hour's shape, or
            # flat if there is no previous hour.
            previous = medians.get(hour - 1)
            medians[hour] = list(previous) if previous is not None else [0.0] * 5
    return medians, lo_all, hi_all


def fallback_log_ratios():
    """Flat persistence forecast (the house baseline's shape)."""
    out = {}
    for hour in range(2, 25):
        spread = FALLBACK_SIGMA * math.sqrt(hour - 1)
        out[hour] = [z * spread for z in Z]
    return out


def forecast_case(case):
    """Forecast one case: {"hours": [{"hour": h, "q": [...]}, ...]}."""
    known = case.get("known") or []
    first = next((k for k in known if k.get("hour") == 1), None)
    p1 = 1.0
    if first is not None:
        try:
            viewers = float(first.get("viewers"))
            if math.isfinite(viewers) and viewers > 0:
                p1 = viewers
        except (TypeError, ValueError):
            pass
    if p1 == 1.0:
        # No usable hour-1 print: use the last pre-checkpoint hourly point.
        hourly = case.get("history_hourly") or []
        for row in hourly:
            try:
                viewers = float(row.get("viewers"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(viewers) and viewers > 0:
                p1 = viewers

    curves = launch_curves(case)
    if len(curves) >= 5:
        log_ratios, lo_all, hi_all = median_log_ratios(curves, math.log(p1))
        log_v1 = math.log(p1)
        outside = max(0.0, log_v1 - hi_all, lo_all - log_v1)
        widen = min(WIDEN_CAP, WIDEN_PER_UNIT * outside)
    else:
        log_ratios = fallback_log_ratios()
        widen = 0.0

    hours = []
    wanted = [row.get("hour") for row in (case.get("forecast") or [])]
    wanted = [h for h in wanted if isinstance(h, int) and 2 <= h <= 24]
    if len(wanted) != 23 or sorted(wanted) != list(range(2, 25)):
        wanted = list(range(2, 25))
    for hour in wanted:
        ratios = log_ratios.get(hour, [0.0] * 5)
        quantiles = []
        for tau_index, ratio in enumerate(ratios):
            # Widen the band, not the median, when extrapolating.
            ratio += (1.0 if Z[tau_index] > 0 else -1.0 if Z[tau_index] < 0 else 0.0) \
                * widen * abs(Z[tau_index]) / 1.6448536269514722
            value = p1 * math.exp(ratio)
            quantiles.append(value)
        # Enforce non-decreasing quantiles and the contest's bounds.
        fixed = []
        running = 1.0
        for value in quantiles:
            running = max(running, value)
            fixed.append(min(1e9, running))
        hours.append({"hour": hour, "q": [round2(v) for v in fixed]})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception:
            # Never leave a case without an output file: a flat
            # forecast at the hour-1 print (or 1 viewer) is valid,
            # even if it is a poor one.
            try:
                p1 = float(next(
                    k["viewers"] for k in (case.get("known") or [])
                    if k.get("hour") == 1 and k.get("viewers"))) or 1.0
            except Exception:
                p1 = 1.0
            forecast = {"hours": [
                {"hour": hour, "q": [round2(p1)] * 5}
                for hour in range(2, 25)]}
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
