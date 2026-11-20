"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Model, in one paragraph: Exophase does not update its trophy page continuously.
The page sits unchanged between refreshes, so a case's history is a sequence of
page *states*, not a smooth series. A rate is a ratio — earners over tracked
players — and its logit trend over log-time is exactly the difference between
the earners' growth and the pool's growth, so dilution (a saturating trophy
falling in percent while the pool keeps growing) is inside the trend rather
than added on. Raw count extrapolation is *not* trusted: an uptake curve's
steep launch phase, projected as a power law, explodes. Instead each trophy's
median is its last rate plus a shrunk, capped log-time slope from a trailing
36-hour fit, and the program backtests itself inside the case — rolling-origin
fits against the last observed state choose between persistence and two trend
strengths, and set the spread from the winning variant's own errors, scaled to
the case's horizon. One regime is handled by name: a history that is still the
frozen pre-launch page (638 players) straddles the first refresh the rules
disclose (3,868 players, earners unchanged), so it is forecast as the same
earners over the refreshed pool.

Runs as `python3 -P -s /bot/main.py` in python:3.13-slim, standard library
only, no network. Outside the grader, MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR
stand in for /input and /output.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]

# Fallback medians (percent) for a trophy with no usable read, by tier.
TIER_PRIOR = {"platinum": 0.4, "gold": 2.5, "silver": 7.0, "bronze": 18.0}


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit_pct(p):
    p = clamp(p, 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(x):
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up, the way the grader's reference rounds."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    v = whole / 10000.0
    return int(v) if float(v).is_integer() else v


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b)."""
    sw = sum(ws)
    if sw <= 0:
        return (ys[-1] if ys else 0.0), 0.0
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return my, 0.0
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    return my - b * mx, b


def extract_states(case):
    """Collapse the read history into distinct page states.

    Returns (states, t_first) where each state is
    (time_hours_from_first_read, players, [rate or None per trophy]).
    Consecutive reads with identical players and rates are one state.
    """
    hist = case["history"]
    t0 = parse_time(hist[0]["at"])
    states = []
    prev_key = None
    for rd in hist:
        key = (rd["players_tracked"], tuple(rd["rates"]))
        if key == prev_key:
            continue
        prev_key = key
        states.append(((parse_time(rd["at"]) - t0) / 3600.0,
                       rd["players_tracked"], list(rd["rates"])))
    return states, t0


def segment_start(states):
    """Index where the current page regime starts: just after the last refresh
    that multiplied the tracked players by more than 1.5x (on launch night the
    refresh took the page from 638 to 3,868 players). Fits and backtests stay
    inside the live segment; the frozen pre-refresh states are another world.
    """
    start = 0
    for k in range(1, len(states)):
        if states[k][1] > 1.5 * states[k - 1][1]:
            start = k
    return start


def trend_median(pts, t_target, shrink, cap):
    """Median rate for one trophy from its live-segment (time, rate) points.

    Anchored at the last observed rate; adds the log-time slope of a weighted
    fit over the trailing 36 hours, shrunk and capped so no median moves more
    than `cap` logit units, and blends a quarter toward the fit's own level at
    the target. The slope of a rate's logit *is* its earners-vs-players growth
    difference, so dilution is inside the trend, not added on.
    """
    last_t, last_r = pts[-1]
    base = logit_pct(last_r)
    if len(pts) >= 3:
        win = [p for p in pts if p[0] >= last_t - 36.0]
        if len(win) < 3:
            win = pts[-3:]
        if win[-1][0] > win[0][0]:
            xs = [math.log(t + 6.0) for t, _ in win]
            ys = [logit_pct(r) for _, r in win]
            n = len(xs)
            ws = [1.0 + (i / max(n - 1, 1)) ** 2 for i in range(n)]
            a, b = wls(xs, ys, ws)
            du = math.log(t_target + 6.0) - math.log(last_t + 6.0)
            move = clamp(b * du, -cap, cap) * shrink
            level = clamp(a + b * math.log(t_target + 6.0), base - cap, base + cap)
            return inv_logit_pct(base + move + 0.25 * (level - base - move))
    return last_r


# (shrink, cap) variants the in-case backtest chooses between; index 0 is
# persistence, which wins whenever the recent trend does not pay for itself.
VARIANTS = [(0.0, 0.0), (0.65, 1.1), (1.0, 1.4)]


def backtest(states, seg, n_tr, t_target):
    """Rolling-origin check inside the case: fit each variant on the live
    segment up to several origins, forecast the segment's last state, and
    measure logit errors. Returns (variant_index, sigma, bias) with sigma
    scaled from the backtest horizons to the case's own horizon.
    """
    seg_states = states[seg:]
    k_all = len(seg_states)
    last_t = seg_states[-1][0]
    origins = []
    for j in range(3, k_all - 1):
        if last_t - seg_states[j][0] >= 6.0:
            origins.append(j)
    if len(origins) > 5:
        step = len(origins) / 5.0
        origins = [origins[int(i * step)] for i in range(5)]
    if not origins:
        return 1, None, 0.0
    errs = [[] for _ in VARIANTS]
    horizons = []
    for j in origins:
        horizons.append(last_t - seg_states[j][0])
        upto = seg_states[:j + 1]
        for i in range(n_tr):
            actual = seg_states[-1][2][i]
            if actual is None:
                continue
            pts = [(s[0], s[2][i]) for s in upto if s[2][i] is not None]
            if not pts:
                continue
            for v, (shrink, cap) in enumerate(VARIANTS):
                pred = pts[-1][1] if v == 0 else trend_median(pts, last_t, shrink, cap)
                errs[v].append(logit_pct(actual) - logit_pct(pred))
    scores = []
    for v in range(len(VARIANTS)):
        if errs[v]:
            scores.append(sum(abs(e) for e in errs[v]) / len(errs[v]))
        else:
            scores.append(float("inf"))
    best = min(range(len(VARIANTS)), key=lambda v: scores[v])
    e = errs[best]
    if not e:
        return best, None, 0.0
    rms = math.sqrt(sum(x * x for x in e) / len(e))
    bias = clamp(0.5 * (sum(e) / len(e)), -0.3, 0.3)
    h_bt = sum(horizons) / len(horizons)
    h_target = max(t_target - last_t, 1.0)
    # Error dispersion grows far slower than a random walk's sqrt: the trend
    # already carries the trajectory, so what remains is mostly level
    # uncertainty. A 0.15 power, validated on simulated launch weeks, keeps
    # the spread honest without paying for width the outcomes do not use.
    sigma = clamp(rms * (h_target / max(h_bt, 1.0)) ** 0.15, 0.08, 0.45)
    return best, sigma, bias


def forecast_case(case):
    trophies = case["trophies"]
    n_tr = len(trophies)
    horizon_days = case["horizon_minutes"] / 1440.0
    states, t0 = extract_states(case)
    t_target = (parse_time(case["target"]) - t0) / 3600.0
    last_t, last_p, last_rates = states[-1]
    target_ts = parse_time(case["target"])

    # Launch-night frozen page: the case straddles Exophase's first refresh.
    # The rules themselves disclose that refresh (638 -> 3,868 players, every
    # rate but the 100% one diluted), so a history that is still the frozen
    # pre-launch page is forecast as the same earners over the refreshed pool.
    if (len(states) == 1 and last_p == 638
            and last_t * 3600.0 + t0 < parse_time("2026-11-19T04:30:00Z")
            <= target_ts < parse_time("2026-11-20T00:00:00Z")):
        medians = []
        for i in range(n_tr):
            r = last_rates[i]
            if r is None:
                medians.append(TIER_PRIOR.get(trophies[i]["tier"], 10.0))
            elif r >= 99.99:
                medians.append(99.9)
            else:
                medians.append(clamp(r * 638.0 / 3868.0, 0.01, 99.99))
        return _rows(trophies, medians, 0.35, 0.0)

    if len(states) == 1:
        medians = [last_rates[i] if last_rates[i] is not None
                   else TIER_PRIOR.get(trophies[i]["tier"], 10.0) for i in range(n_tr)]
        return _rows(trophies, medians, max(0.30 * math.sqrt(max(horizon_days, 0.02)), 0.10), 0.0)

    seg = segment_start(states)
    if len(states) - seg < 4:
        seg = 0
    variant, sigma_bt, bias = backtest(states, seg, n_tr, t_target)
    sigma = sigma_bt if sigma_bt is not None else max(0.30 * math.sqrt(max(horizon_days, 0.02)), 0.10)
    shrink, cap = VARIANTS[variant]

    medians = []
    for i in range(n_tr):
        pts = [(s[0], s[2][i]) for s in states[seg:] if s[2][i] is not None]
        if not pts:
            medians.append(TIER_PRIOR.get(trophies[i]["tier"], 10.0))
            continue
        if variant == 0:
            medians.append(pts[-1][1])
        else:
            medians.append(trend_median(pts, t_target, shrink, cap))
    return _rows(trophies, medians, sigma, bias, last_rates)


def _rows(trophies, medians, sigma, bias, last_rates=None):
    rows = []
    for i, trophy in enumerate(trophies):
        med = clamp(medians[i], 0.01, 99.99)
        s = sigma
        if last_rates is not None and last_rates[i] is not None:
            if last_rates[i] >= 99.99:
                s = sigma * 0.5
                med = min(med, 99.95)
            elif last_rates[i] <= 0.05:
                s = sigma * 1.2
        lm = logit_pct(med) + bias
        qs = [round4(clamp(inv_logit_pct(lm + z * s), 0.01, 99.99)) for z in Z]
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        if case["levels"] != LEVELS:
            raise ValueError(f"{path.name} asks for levels {case['levels']}")
        out = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
