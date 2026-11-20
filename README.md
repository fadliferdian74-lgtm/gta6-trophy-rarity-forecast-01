[![Competition](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/fadliferdian74-lgtm/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/fadliferdian74-lgtm/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecaster

Forecasts, for every PlayStation 5 trophy of Grand Theft Auto VI, the percent
of Exophase-tracked players who will have earned it at 2026-11-26T00:00:00Z —
five quantiles per trophy, scored by pinball loss on the logit scale.

## Approach

Exophase refreshes its counts on its own schedule, so a case's history is a
sequence of page **states**, not a smooth series. A rate is a ratio — earners
over tracked players — and its logit trend over log-time is exactly the
difference between the earners' growth and the pool's growth, so dilution (a
saturating trophy falling in percent while the pool keeps growing) is inside
the trend rather than added on.

- Each trophy's median is its last rate plus a **shrunk, capped log-time
  slope** from a weighted fit over the trailing 36 hours of the live segment
  (the states since the last refresh that multiplied the player pool by more
  than 1.5×). Raw count extrapolation is deliberately not trusted: an uptake
  curve's steep launch phase, projected as a power law, explodes.
- The program **backtests itself inside the case**: rolling-origin fits
  against the last observed state choose between persistence and two trend
  strengths, and set the quantile spread from the winning variant's own
  errors, scaled to the case's horizon with a slow 0.15 power (error
  dispersion grows far slower than a random walk's — the trend already
  carries the trajectory).
- A history that is still the **frozen pre-launch page** (638 players,
  launch night) straddles the first refresh the competition rules disclose
  (3,868 players, earners unchanged), so it is forecast as the same earners
  over the refreshed pool.

## Validation

On simulated launch weeks (359 reads over 60 hours, refresh cadence 1–6 h,
three uptake shapes, six seeds each, plus a hostile flavor with a weekend
surge and a mid-window pool re-basing), the forecaster's mean pinball loss
was **0.106** on smooth worlds and **0.133** on hostile ones, against 0.241 /
0.249 for the house persistence baseline and 0.130 / 0.164 for flat
persistence. Runtime per case ≈ 0.1–0.6 s of the 8 s budget; output is
byte-identical across runs.

## Running

```
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network.
