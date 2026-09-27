# Expert Replay Lessons & Empirical Playbook (Kaggriculture MuZero)

Empirical synthesis distilled from Kaggle competition episode replays in `replays/`, representing top leaderboard scores ranging from **$80,000 to $147,749**.

---

## 1. Top Replay Performance Overview

| Replay ID | Seat | Score | Status | Key Strategic Distinctives |
| :--- | :---: | :---: | :---: | :--- |
| `113532514.json` | 0 | **$154,934** | Winner (Maxy5illion) | All-time highest score: Day 6 H7 NE, Day 11 H2 SW ($15.4k liquidity), SE skipped, 0 Tomato, 12 hands, 12 Melon, 33 Strawberry, 31 Carrot, 8 Cows, 9 Sheep. |
| `113532514.json` | 1 | **$154,932** | Runner-up (T0m0t0m0 W) | Near-perfect mirror match execution within $2 delta of winner. |
| `113554880.json` | 0 | **$148,239** | Winner (Akshay Mathur) | NE Day 6 H7, SW Day 11 H2 ($16.2k cash), SE skipped, 11 hands, 9 Cows, 5 Sheep, 3 Geese. |
| `113554880.json` | 1 | **$148,113** | Runner-up (Kamilek) | High-margin milk/wool cashflow, symmetric elite performance. |
| `113514650.json` | 1 | **$147,749** | Winner (Boey) | SW Day 8, SE skipped, 0 Tomato, 12 hands, massive Cow + Sheep milk & wool cashflow. |
| `113370862.json` | 1 | **$141,034** | Winner (Boey) | SW Day 9, SE expanded Day 18, 14 Tomato seeds, 13 hands, elite livestock scaling. |
| `113497352.json` | 0 | **$136,022** | Winner | SW Day 8, SE skipped, 20 Sheep, 8 Cows, 4 Geese, 12 hands. |
| `113524956.json` | 1 | **$134,115** | Winner (Scott Weeden) | SW Day 11 H2 ($16.1k cash), SE skipped, 12 hands, 9 Cows, 5 Sheep, 3 Geese. |
| `113317026.json` | 1 | **$132,913** | Winner (Majkel1337) | SW Day 6, SE expanded, 30 Tomato seeds, 12 hands, continuous shed liquidation. |
| `113512450.json` | 0 | **$132,501** | Winner (TheEggman) | SW Day 6, SE expanded, 20 Tomato seeds, 12 hands. |
| `113560261.json` | 1 | **$132,218** | Winner (Max Power) | SW Day 9, SE skipped, 0 Tomato, 11 hands, high-margin livestock. |
| `113513018.json` | 1 | **$131,407** | Winner (Scott Weeden) | Step 1 NeML interleaved wheat arb, early Cow+Sheep, Melon harvest explosive cashflow, 33 Strawberry, 0 Tomato, SW Day 8/11, SE skip. |

---

## 2. Core Strategic Formula

Across all matches scoring ≥ $115,000, the following rules were universally observed:

### A. Turn 0 / 1 Opening Arbitrage (The NeML Sequence)
- **Turn 0 (Hour 0)**: PASS on primitive actions; market is evaluated.
- **Turn 1 (Hour 1)**: Execute lockstep interleaved product transactions:
  1. `["BUY_PRODUCT", "WHEAT", 9]`
  2. `["SELL", "WHEAT", 9]`
  3. `["BUY_PRODUCT", "WHEAT", 11]`
  4. `["SELL", "WHEAT", 6]`
  5. `["BUY_SEED", "WHEAT", 1]`
  - *Effect*: Extracts positive cash delta (+$20–$25) against standard bulk buyers without taking inventory or price risk.
- **Turn 2 (Hour 2)**: Immediately convert cash into **5 Hands** (`5x HIRE`) and livestock:
  - `["BUY_ANIMAL", "COW", 2]`
  - `["BUY_ANIMAL", "SHEEP", 2]`

### B. Crop Portfolio Constants
1. **Melon (12 seeds total)**:
   - Sown early in NW/NE.
   - Harvests around **Day 10** in multiple staggered waves (6, 12, 12, 6 units).
   - Generates **$12,000 to $16,000 in immediate cash**, creating the critical liquidity bridge for SW expansion and livestock ramp.
2. **Strawberry (exactly 33 seeds total)**:
   - Purchased continuously in single-seed increments from Day 7 through Day 25.
   - Sold on high-value morning liquidation hours (hours 1–2).
3. **Carrot (28 to 40 seeds)**:
   - Modest carrot cultivation post-NE unlock; serves as auxiliary cashflow without clogging shed capacity.
4. **Tomato (0 seeds in top 131k; 10–21 seeds only when expanding SE)**:
   - The top 2 scores ($131.4k and $130.8k) completely avoided Tomato.
   - Only useful when SE quadrant is unlocked around Day 18 to utilize spare tile acreage.

### C. Livestock & Animal Management
- Recurring animal produce (**Wool, Milk, Eggs**) provides sustainable high-margin income that is impervious to soil fatigue:
  - **Cows (6 to 12)**: Yields recurring Milk (sold at ~$256/unit).
  - **Sheep (5 to 6)**: Yields recurring Wool (sold at ~$240/unit).
  - **Geese/Chickens (3 to 5)**: Yields Eggs/Feathers.
- Feed stock: Buy WHEAT products in small batches during low-price hours to keep animals producing without exceeding shed inventory buffers.

### D. Land Expansion Discipline
- **NE Quadrant**: Always purchased on **Day 6, Hour 6–7** ($1,000 price + buffer).
- **SW Quadrant**:
  - Economically affordable as early as Day 8 Hour 4 ($2,500 buffer).
  - Executed between **Day 8 and Day 11 Hour 2** (post-melon liquidity explosion).
- **SE Quadrant ($4,000)**:
  - **Skip SE by default**: 11–12 hands distributed across NW, NE, and SW maximizes tile action density without incurring excessive daily wage burn.
  - **Late SE condition**: Only expand to SE if cash exceeds $30,000–$40,000 around Day 18 and hand count can be scaled to 14+.

### E. Market & Shed Liquidation
- **Liquidation Hours**: Hours **1 and 2** daily (and continuously after Day 27).
- **Shed Force Sell**: When non-seed shed items reach **≥ 85 items**, force sell surplus down to **75 items**, prioritizing highest value per unit (Strawberry, Milk, Wool, Melon).
- **Phantom Sell Clamp**: Strictly clamp sell order quantities to actual verified shed stock (`_clamp_sells`).

---

## 3. MuZero Architecture & Integration

The learned MuZero stack operates as a **macro-decision gate** grafted onto the Hybrid Direct chassis:

1. **State Vector ($o_t \in \mathbb{R}^{32}$)**:
   - Log capital $\ln(1 + \text{money})$, normalized day ($t/30$), unlocked quadrant flags, shed inventory per SKU, and spot-to-base price ratios $P_t / P_0$.
2. **Macro Options**:
   - `0: FRONT_LOAD_SURVIVAL` (days $\le 6$)
   - `1: EXPANSION_PREPARATION` (cash accumulation)
   - `2: TRICKLE_LIQUIDATION` (spot premium selling)
   - `3: EXPAND_NE`, `4: EXPAND_SW`, `5: EXPAND_SE`
   - `6: HIRE_HANDS`
   - `7: PASS`
3. **MCTS Planning**:
   - Gumbel-Top-$k$ sequential halving with 12–16 simulations per evaluation within a 45ms timeout.
   - Evaluates whether SE expansion yields net terminal wealth delta $> 0$ after accounting for labor cost curves.

---

## 4. Multi-Phase Training Pipeline

```
Replays (replays/*.json)
   │
   ├─► learn_from_replay.py --all
   │     └─► artifacts/ep*_policy.json (per-episode constants)
   │
   ├─► train_muzero.py
   │     ├─► Phase 1: Analytical bootstrap (O(1) economy transitions)
   │     ├─► Phase 2: MCTS self-distillation (policy target reanalyse)
   │     └─► Phase 3: Expert BC on top trajectories (score >= 80k)
   │           └─► artifacts/muzero_checkpoints.pt
   │
   └─► scripts/build_muzero_on_hybrid.py --with-weights
         └─► dist/main.py & dist/submission.zip
```

### Empirical Validation
In head-to-head tournament games against all 5 local baselines with rotating seeds and alternating player seats (P0 ↔ P1):
- **vs submission_2000**: 5-1 (83.3% win rate, +$5,670 avg delta)
- **vs submission_2500**: 6-0 (100.0% win rate, +$4,675 avg delta)
- **vs submission_hybrid_direct**: 6-0 (100.0% win rate, +$484 avg delta)
- **vs submission_muzero**: 5-1 (83.3% win rate, +$687 avg delta)
- **vs submission_thirst**: 6-0 (100.0% win rate, +$109,368 avg delta, scoring up to $173,216)
- **Overall**: **28-2-0 (93.3% win rate)** across 30 fair matches.

---

## 5. Live Match Case Study: Episode 113565586 (`Scott Weeden` vs `Thirst`)

A head-to-head live Kaggle match revealing four key tactical differences in mid-to-late game resource allocation:

- **Seat 0**: `Scott Weeden` — **$64,803**
- **Seat 1**: `Thirst` — **$78,954** (**+$14,151 delta / Winner**)

Both players were virtually tied through Day 10 ($15,587 vs $14,781). The +$14,151 divergence was driven by three specific execution edges:

### 1. Late-Game Carrot Surge (Days 20–26) — *+$8,653 Revenue Delta*
- **Mechanism**: Carrots mature in exactly 3 game days. Seeds planted between Day 20 and Day 26 can be 100% harvested and liquidated before the Day 30 match end.
- **Thirst**: Scaled up to **125 Carrot seeds** (planting 13–19 daily across Days 20–26), harvesting **323 Carrots** for **$23,198** in revenue.
- **Scott Weeden**: Planted only 68 Carrot seeds, harvesting 205 Carrots for $14,545.

### 2. Auxiliary Poultry Monetization (2 Geese) — *+$3,734 Revenue Delta*
- **Thirst** purchased **2 Geese**, harvesting and selling **68 Eggs** for **$3,734**.
- Because animal feed (Wheat) is pooled, feeding 2 Geese added near-zero marginal overhead while unlocking a high-margin third product line. Scott purchased 0 Geese ($0 revenue).

### 3. Second-Wave Melon Injection (Day 10) — *+$2,802 Revenue Delta*
- Immediately upon receiving the initial Day 10 Melon harvest liquidity surge, Thirst re-invested into **4 replacement Melon seeds** (15 total seeds vs Scott's 12), yielding **90 Melons ($15,402)** vs Scott's 72 Melons ($12,600).

### 4. The Fertilizer Labor Trap
- Scott gathered and sold **2,335 Fertilizer units** for $17,584 (an average spot price of only $7.52/unit), burning **318 `COLLECT_FERTILIZER`** and 81 `FERTILIZE` hand actions.
- Thirst reallocated those hand actions directly into **271 `PLANT` actions** (+44 over Scott) and watering.
- *Lesson*: In late stages, worker actions have high opportunity cost. Hands should prioritize planting high-margin cash crops over grinding low-value compost.


