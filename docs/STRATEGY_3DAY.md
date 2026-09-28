# Kaggriculture: the 3-day plan (rethink of the MuZero approach)

_Status: draft written on day 1; numbers below come from local games with
kaggle-environments 1.32.7 and the agents checked into this repo._

## 1. What the game actually is

Kaggriculture is a two-player, 720-turn farming game whose **only coupling
between the players is the shared market**. Each player's farm is private
except for its public tiles; the market book (inventory and price per product),
the town's unlocked shops and both farms' tiles are public. Final score is cash.

Three facts drive everything below.

| Measurement (seed 10101 unless noted) | Result |
| :--- | ---: |
| Champion chassis vs a passive opponent | $163k |
| Champion chassis vs itself (mirror) | $93k each |
| Champion mirror across 8 seeds (11-18) | $72k to $149k |
| Pure MuZero submission (`main_mz.py`) vs champion | $2.4k vs $187k |
| Modular replay chassis (`main_pl.py`) vs champion, 8 paired games | 0 wins, $72k vs $118k |

* The market is worth about as much as the farming: the same farm plan earns
  roughly half as much when a second copy of it sells into the same book.
* The town's random shop draw (which products the town consumes) moves the
  mirror score by a factor of two. It is public from day 3 onward.
* MuZero added nothing. It learned a dynamics model for an environment whose
  dynamics are 1,000 lines of public Python that the submission can simply
  include. Its sampled-MCTS build does not even reach the opening.

## 2. What the champion is

`baselines/champion_main_20260927_051846.py` is a **shop-router**: 41
pre-recorded 719-step action tapes (all variants of one plan: 12 melon, 33
strawberry, 31 carrot, 163 wheat, 6-8 cows, 6-11 sheep, 0-5 geese, 11-12
hands, NE on day 6, SW on day 11, SE skipped), a router that picks a tape on
day 6 from the first two shops, and ~50 reactive guard layers stacked on top
(sell clamps, weed repair, courier, fertilizer use, carrot-for-wheat swaps,
two species-swap layers with hand-tuned gates, rival sale prediction).

It is strong because the tapes were distilled from top public replays. It is
brittle because it is open-loop: in a mirror match both players plant and sell
the same things at the same time and crash the same books.

## 3. The lever: multi-agent, yes; neural world model, no

The competitor population is mostly derived from the same public lineage, so
the rival's production is near-deterministic and largely visible (tiles show
every animal and crop; market inventory deltas reveal every sale). That makes
this a **best-response problem with a known model**, and the practical
"multi-agent RL" toolkit for three days is:

1. **League evaluation** (this repo's `arena/`): paired seeds, both seats,
   process-isolated games, exact per-product revenue attribution. Roughly 400
   games per core-hour, so a candidate can be judged on 16-32 paired games in
   minutes.
2. **Exact market model** (`strategy/market_model.py`): the engine's price
   curves, shop consumption and animal/crop production schedules, verified
   against the engine. Demand from shops not yet drawn is added in expectation.
3. **Planning layers on the champion** (`strategy/layers/`): decisions the
   tape makes blindly are re-decided at decision time from the market model,
   the shop draw and the rival's visible farm. Each layer is appended in the
   chassis' own wrapping style and compiled to a single file by
   `strategy/build.py`.
4. **Population-based tuning**: every layer exposes its dials as module
   globals so the arena can sweep them (`--a-overrides '{"_SP_MIN_GAIN": 0}'`)
   against a pool of opponents (champion, its tape variants, `main_pl.py`, and
   previous candidates) rather than a single fixed rival.

What is deliberately out of scope: training a policy from scratch (a 720-step
game with a combinatorial action space, no GPU, three days), and the
MuZero/LoRA/PPO pipelines, which are left in place but unused.

## 4. Day-by-day

**Day 1 (this branch)**: arena + market model + species planner; measure the
headroom of species choice with forced-cow / forced-sheep variants; gate the
EV planner against the champion on paired seeds.

**Day 2**: sell-timing layer (hold a product while the rival is dumping it and
consumption will restore the price; release before the end), late-carrot
sizing from remaining carrot capacity, rival-sale recovery from inventory
deltas; sweep dials against the opponent pool; build and submit candidate v1.

**Day 3**: robustness (timeouts, exceptions, seat/seed coverage), final sweep,
submit the best gated candidate, keep the previous best as the fallback.

## 5. What I need from you

* The 50 gold replays (`replays/*.json`, git-ignored) and the scored agents in
  `/Volumes/BASELINES`: neither volume is mounted in this cloud container. A
  branch with `git add -f` of the replays, or a tarball link, lets me build a
  realistic opponent pool and validate the market model against real matches.
* Your current leaderboard rating and the top score, so the gate threshold
  matches what a rating gain actually needs.
