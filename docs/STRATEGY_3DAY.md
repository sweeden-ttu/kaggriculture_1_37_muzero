# Kaggriculture: the 3-day plan (rethink of the MuZero approach)

_Day-1 write-up. Every number below comes from local games with
kaggle-environments 1.32.7 and the agents checked into this repo, played with
the arena in `arena/`._

## 1. What the game actually is

Kaggriculture is a two-player, 720-turn farming game whose **only coupling
between the players is the shared market**. Farms are private except for their
public tiles; the market book (inventory and price per product), the town's
unlocked shops and both farms' tiles are public. Final score is cash.

| Measurement | Result |
| :--- | ---: |
| Champion chassis vs a passive opponent (seed 10101) | $163k |
| Champion chassis vs itself (mirror, seed 10101) | $93k each |
| Champion mirror across 12 seeds | $72k to $156k |
| Pure MuZero submission (`main_mz.py`) vs champion | $2.4k vs $187k |
| Modular replay chassis (`main_pl.py`) vs champion, 8 paired games | 0 wins, $72k vs $118k |

* The market is worth about as much as the farming: the same farm plan earns
  roughly half as much when a second copy of it sells into the same book.
* The town's random shop draw (which products the town consumes) moves the
  mirror score by a factor of two. It is public from day 3 onward.
* MuZero added nothing. It learned a dynamics model for an environment whose
  dynamics are 1,000 lines of public Python that the submission can include.
  Its sampled-MCTS build does not even reach the opening.

## 2. What the champion is, and how tight it is

`baselines/champion_main_20260927_051846.py` is a **shop-router**: 41
pre-recorded 719-step action tapes (all variants of one plan: 12 melon, 33
strawberry, 31 carrot, 163 wheat, 6-8 cows, 6-11 sheep, 0-5 geese, 11-12
hands, NE on day 6, SW on day 11, SE skipped), a router that picks a tape on
day 6 from the first two shops, and ~50 reactive guard layers stacked on top
(sell clamps, weed repair, courier, fertilizer use, carrot-for-wheat swaps,
two species-swap layers with hand-tuned gates, rival sale prediction).

Measured on a full game (seed 13):

* Land: 25/25 tiles used from day 1, 50/50 from day 8, 75/75 from day 13
  to day 27.
* Labour: 250-275 unit-actions per day from day 10, of which 0-25 are PASS
  (2-10% slack) and about 40% are moves.
* Cash: $300-$1,100 from day 5 to day 8 with every dollar committed to seeds,
  cows and the NE purchase; diverting $500 on day 6 fails the day-7
  strawberry seeds.
* Waters ongoing crops every other day (the engine only kills a plant after
  two consecutive dry days) and already fertilizes strawberries (62
  applications per game).

It is strong because the tapes were distilled from top public replays. It is
brittle because it is open-loop: in a mirror match both players plant and sell
the same things at the same time and crash the same books.

## 3. Levers tested on day 1 (and why most are small)

| Lever | Result (paired seeds, both seats, vs champion) |
| :--- | :--- |
| Force every pasture animal to sheep | -$14k per game |
| Force every pasture animal to cow | -$160 mean, but -$7k to +$9k per seed |
| Species re-decided per purchase by price-path EV (`strategy/layers/species_planner.py`) | +$500 mean, 4W-8L-4T: the router already conditions on the draw, and the EV at day 9 cannot see the day-12 shop |
| Melon front-run (fertilize on day 6, harvest day 9, sell before the rival's day-10 dump) | **Impossible**: the engine refuses HARVEST before `first_yield_day` (10 for melon) whatever the yield; the tape already sells at the first legal hours |
| Extra dedicated hand | Feasible only on days whose tape hires are all at hour 0 (a later tape hire shifts hand indices); hand 13 costs $233/day, hand 14 $377, so extra labour does not pay |

The first dial sweep (20 sell/swap/carrot dials of the champion, 7 random
configurations, 16 paired games each against the champion itself) produced
deltas of -$117 to +$60 with a standard error of about $110: **against an
identical rival those layers never trigger**, and games with different sell
dials were byte-identical. Overrides themselves work (an extreme carrot-swap
setting moved the score by $19k). Tuning therefore needs an opponent pool with
real differences in mix and timing: `dist/pool/` holds the champion pinned to
tapes 0, 12, 100 and 110 (`strategy/layers/route_force.py`), an all-cow
variant and `main_pl.py`.

Three evaluation facts matter more than any single lever:

1. **The shop draw is coupled to the farms.** The engine seeds one RNG per day
   and consumes one draw per empty tile on both farms for weeds before drawing
   the day's new shop, so any change in either farm re-rolls every later shop.
   The same seed gave a yarn store in one game and three bakeries in another.
   Paired seeds therefore do not control the draw; `arena --fixed-draw`
   decouples it (common random numbers) so candidates can be compared.
   In real games the coupling is a zero-mean re-roll, not an exploit: the seed
   is not observable.
2. **Seat asymmetry is small** (about $900 on some seeds from P0 acting first
   in atomic orders), so both seats must be played but seat is not a lever.

## 4. The plan that survives contact

The population is mostly derived from the same public lineage, so the rival's
production is near-deterministic and largely visible. This is a best-response
problem with a known model, and the practical "multi-agent RL" toolkit for
three days is:

1. **League evaluation** (`arena/`): paired seeds, both seats, fixed draw,
   process-isolated games, exact per-product revenue attribution (reconciles to
   the final money to the dollar). About 400 games per core-hour.
2. **Exact market model** (`strategy/market_model.py`): the engine's price
   curves, shop demand and production schedules, verified against the engine.
3. **Population-based dial tuning** (`arena/sweep.py`): every tunable in the
   champion is a module global, so the arena sweeps them against an opponent
   pool without editing the 7,600-line file (random search, then CEM). The
   first sweep runs the champion's own 20 sell/swap/carrot dials against the
   champion; its best configuration is re-validated on held-out seeds with the
   fixed draw before it becomes a candidate.
4. **Opponent pool, not a single rival**: the champion, its tape variants,
   `main_pl.py`, previous candidates, and, as soon as they are available, the
   scored agents from `/Volumes/BASELINES` and tapes cut from the 50 gold
   replays. Tuning against a mirror alone overfits.

Out of scope: training a policy from scratch (a 720-step game with a
combinatorial action space, no GPU, three days), and the MuZero/LoRA/PPO
pipelines, which stay in the repo but are unused.

## 5. Day-by-day

**Day 1 (this branch)**: arena, fixed draw, attribution, market model, species
planner (parked, `_SP_MODE="tape"`), sweep infrastructure and first sweep.

**Day 2**: sweep against a pool (champion + `main_pl.py` + baseline agents),
held-out validation, a sell-timing layer that uses recovered rival sales and
the consumption forecast (hold a product while the rival is dumping it and
consumption will restore the price; release before the end), build and submit
candidate v1.

**Day 3**: robustness (timeouts, exceptions, seat and seed coverage), final
sweep, submit the best gated candidate, keep the previous best as fallback.

## 6. What I need from you

* The 50 gold replays (`replays/*.json`, git-ignored) and the scored agents in
  `/Volumes/BASELINES`: neither volume is mounted in this cloud container.
  A branch with `git add -f` of the replays, or a tarball link, lets me build a
  realistic opponent pool and validate the market model against real matches.
* Your current leaderboard rating and the top score, so the gate threshold
  matches what a rating gain actually needs.

## 7. How to run

```bash
# paired-seed match, both seats, fixed draw, per-product attribution
python -m arena.run --a dist/candidate.py --b baselines/champion_main_20260927_051846.py \
    --seeds 1-16 --fixed-draw --attribution --save artifacts/arena/cand_vs_champ.json

# dial sweep against a pool
python -m arena.sweep --agent baselines/champion_main_20260927_051846.py --dials arena/dials_v1.json \
    --opponents baselines/champion_main_20260927_051846.py dist/main_pl.py \
    --seeds 11-18 --trials 200 --fixed-draw --log artifacts/arena/sweep.jsonl

# build a candidate = champion + layers (+ build-time dial overrides)
python -m strategy.build --layers species_planner --out dist/candidate.py --set "_SP_MODE='tape'"
```
