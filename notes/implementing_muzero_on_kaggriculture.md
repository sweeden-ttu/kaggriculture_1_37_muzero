# Implementing MuZero on Kaggriculture

Practical notes distilled from `papers/` and mapped onto this
repo’s Dec-POMDP (720 turns, 4 quadrants, market cap 10, shed cap 100).

Companion index: [`../README.md`](../README.md)  
Code today: `muzero.py`, `macro_mcts.py`, `train_muzero_offline.py`,
`expansion_policies.MuZeroExpansionPolicy`.

---

## 1. What the papers say (and what Kaggriculture needs)

### 1.1 Classic MuZero (Schrittwieser et al., `00_…pdf`)

Learn three functions and plan with MCTS **inside** the learned model:

| Net | Paper role | Kaggriculture mapping |
| :--- | :--- | :--- |
| \(h_\theta\) | \(o_t \to s^0\) | 32-d farm economy vector → latent \(s\in\mathbb{R}^{32}\) |
| \(g_\theta\) | \((s,a)\to(r,s')\) | Macro-option → next-day economy (or analytical hybrid) |
| \(f_\theta\) | \(s\to(\pi,v)\) | Prior over 8 macro options + net-worth value |

**Do not** plan over farmer/hands tile primitives (≈42×42 × chore vocab). That
tree is intractable within `actTimeout`. Papers that succeed on board games have
action branching tens–hundreds; Atari has ≤18. Kaggriculture’s *primitive* space
is far larger → **macro options are mandatory** (same abstraction argument as
Sampled MuZero / ARCHITECTURE_REVIEW).

### 1.2 Value equivalence (`14_…`, `14b_…`)

MuZero does **not** need to predict full next tiles/prices. It only needs a model
that is **value-equivalent** for the policies you plan with: correct enough
Bellman backups for \(v\) and \(\pi\).

Implication for farming:

- Prefer features that drive net worth: money, day, unlocks, shed, relative
  prices, land headroom — not raw tile bitmaps.
- Reward = scaled Δ terminal-ish wealth (`REWARD_SCALE = 1/1000`), not step
  cosmetic churn.
- Hybrid dynamics (`AnalyticalEconomicModel`) are a legitimate value-equivalent
  \(g\) while \(f_\theta\) training catches up (this repo’s default).

### 1.3 Reanalyse / Unplugged (`03_…`)

Reuse stored trajectories; recompute improved \(\pi,z\) with the **current**
network + MCTS (no new env steps). On Kaggriculture:

- Offline BC from expert zips / live episodes ≈ cheap Unplugged.
- After policy changes, **re-label** stored MacroStates with fresh MCTS π
  (Phase 2 distill in `train_muzero_offline.py`).
- Never require on-policy Kaggle interactions to improve weights.

### 1.4 EfficientZero / V2 (`04_…`, `16_…`)

Sample efficiency tricks that transfer:

| Trick | Use on Kaggriculture? |
| :--- | :--- |
| Self-supervised latent consistency \(s_{t+1}\approx g(h(o_t),a)\) | Yes — stabilize hybrid→learned \(g\) transition |
| Value prefix (predict cumulative reward prefix) | Optional; day-scale rewards are sparse |
| Off-policy correction via model | Useful once replaying old expert logs |

### 1.5 Sampled MuZero (`05_…`)

For huge/continuous action spaces, sample a subset of actions for search +
policy targets. On Kaggriculture:

- Macro space is only **8** options → full enumeration is fine.
- If you later expand to “which crop / which animal SKU”, sample 8–16 candidates
  per node instead of the full shop catalog.

### 1.6 Gumbel MuZero (`09_…`)

Root policy improvement with Gumbel-Top-k + Sequential Halving when simulations
are few. **Highly relevant**: Kaggle `actTimeout≈1s` ⇒ maybe 8–32 sims, not 800.

Recommendation: Gumbel root when `n_simulations ≤ 16`; keep plain P-UCT deeper
in the tree.

### 1.7 Stochastic MuZero (blog notes)

Shared market prices + opponent = **stochastic / partially observed** from each
seat. Full Stochastic MuZero (afterstates + chance nodes) is the long-term fit.
Near-term approximation already in use:

- Treat opponent + price path as noise absorbed into value targets.
- Or sample 2–4 price-shift “chance” outcomes at chance nodes later.

### 1.8 UniZero / ReZero / LightZero (`11_…`–`13_…`)

- **UniZero**: transformer history over days — useful if you condition on multi-
  day price traces; overkill for 32-d day-state v1.
- **ReZero**: faster reanalyse — use when expert buffers get large.
- **LightZero**: reference implementations of the whole family (PyTorch).

---

## 2. Kaggriculture Dec-POMDP → MuZero design

### 2.1 Time base

| Engine | MuZero step |
| :--- | :--- |
| 720 turns (30×24) | Prefer **daily** macro decisions (30 planning steps) |
| Hourly micro chores | Heuristic / Rainbow / Hybrid chassis — **outside** MuZero tree |

MuZero owns: land, hire intensity, liquidation regime.  
Chassis owns: pathing, water/feed/harvest, ≤10 market orders.

### 2.2 Macro action set (current)

```
0 FRONT_LOAD_SURVIVAL
1 EXPANSION_PREPARATION
2 TRICKLE_LIQUIDATION
3 EXPAND_NE   4 EXPAND_SW   5 EXPAND_SE
6 HIRE_HANDS
7 PASS
```

Legal mask: only next-in-sequence expand; SE needs extra capital buffer
(`META_SE_EXTRA_BUFFER`); no expand after day ~24.

### 2.3 Observation encoding (32-d)

Already in `encode_macro_state` / `encode_observation`:

- \(\log(1+\text{money})\), day/30, unlock one-hots + count  
- shed totals + key crops/products  
- relative spot prices \(P_t / P_0\)  
- land headroom, liquidation flags (day≥27, shed≥70), stage flags (≤8/18/23)

Extend later (still keep dim modest): hand count, fib hire tier, rival money if
visible, predicted price-shift from `AdversarialMarketPredictor`.

### 2.4 Reward

```
r = (wealth(s') - wealth(s)) * REWARD_SCALE
wealth ≈ capital + analytical liquidation of shed
```

Aligns with tournament score (final money) better than sparse BUY_LAND bonuses.

### 2.5 Hard engine invariants (never break in action decoders)

- Market orders ≤ **10** / turn  
- Shed ≤ **100** (emergency dump ≥90 → &lt;75)  
- Fibonacci hire costs only as cost table  
- No CFR / dual-game resurrected into the policy path  

---

## 3. Reference architecture (what to build / what exists)

```
obs (Kaggle)
  → encode_observation → o ∈ R^32
  → h_θ → s0
  → Latent MCTS over MacroOption (P-UCT or Gumbel root)
        dynamics: AnalyticalEconomicModel  [hybrid]
               or g_θ(s,a)                 [learned]
        prior/value: f_θ
  → best macro option
  → MultiHemisphereRLAgent / Hybrid chassis executes farmer+hands+market
       MuZeroExpansionPolicy gates BUY_LAND
       optional: inject HIRE / liquidation regime from option
```

**Exists today**

- Networks + hybrid MCTS + planner: `muzero.py`  
- Analytical transitions: `macro_mcts.py`  
- Offline Phase1/2/3 trainer: `train_muzero_offline.py`  
- Policy hook: `MuZeroExpansionPolicy` in `expansion_policies.py`  
- Checkpoint: `muzero_checkpoints.pt`  

**Still weak vs papers / ladder**

- MuZero only **gates expansion** strongly; hire/liquidation soft  
- No Gumbel root; sim counts modest  
- Learned \(g_\theta\) underused (hybrid on by default)  
- No Stochastic chance nodes for market  
- Expert BC from Hybrid/Harvest-ledger helps π but chassis still dominates score  

---

## 4. Implementation recipe (ordered)

### Phase A — Keep hybrid dynamics, train \(f_\theta\) hard (now)

1. Freeze or lightly train \(h_\theta\); rely on `AnalyticalEconomicModel` for
   rollouts (value-equivalent shortcut).  
2. Phase 1: `muzero_bootstrap_loss` on synthetic day transitions.  
3. Phase 2: MCTS distill — regress π→visit counts, v→root/wealth blend.  
4. Phase 3: BC from strong experts (Hybrid / Harvest Ledger / own MuZero zip)
   via `collect_expert_transitions_from_zips`.  
5. Eval **only** with `tournament_sub_vs_base.py` + pinned `--seed`.  

### Phase B — Wire options into the live agent (careful)

Stateful Hybrid/Harvest chassis **desync** if you rewrite their market mid-turn
arbitrarily. Safe pattern:

- MuZero chooses option at **day boundaries** (hour 0) or expansion checks only.  
- `MuZeroExpansionPolicy.evaluate_expansion` already does confirm-or-deny.  
- For hire: bump target hands when option=`HIRE_HANDS`, never invent illegal
  fib costs.  
- For liquidation: flip existing liq-hour / dump thresholds, don’t invent parallel
  order books.

### Phase C — EfficientZero-style stability

1. Consistency loss: \(\|h(o_{t+1}) - g(h(o_t),a_t)\|_2\) on real episode pairs.  
2. Keep hybrid rollouts until consistency &lt; threshold, then mix learned \(g\).  
3. Categorical value support optional once returns span large $ ranges.

### Phase D — Timeout-safe search (Gumbel)

1. Budget sims by wall clock (e.g. stop at 5ms remaining).  
2. Gumbel-MuZero root with \(m\in\{4,8,16\}\).  
3. Eager-load networks at **import** time (same lesson as secure bootstrap:
   never pay cold start inside `actTimeout`).

### Phase E — Stochastic prices / opponent

1. Chance outcomes = discretized price-shift buckets or “rival dump / quiet”.  
2. Afterstate = post-own-market, pre-rival/town tick.  
3. Only worth it after Phases A–D beat Minimax/MacroMCTS baselines.

---

## 5. Training loop (concrete)

```text
buffer ← episodes from env or expert zips
for update in 1..N:
  batch ← sample K-step windows (prefer day-aligned)
  s0 ← h(o0)
  for k in 0..K-1:
    p_k, v_k ← f(s_k)
    r_k, s_{k+1} ← g(s_k, a_k)     # or analytical step + h(o_{k+1})
  L ← CE(p, π*) + ℓ_v(v, z) + ℓ_r(r, u) [+ ℓ_consistency]
  # π*, z from MCTS reanalyse or expert labels
```

Targets:

| Symbol | Source |
| :--- | :--- |
| \(\pi^*\) | MCTS visit softmax or expert MacroOption one-hot |
| \(z\) | n-step / terminal money / analytical wealth |
| \(u\) | Observed Δmoney or analytical Δwealth |

Hyperparams that already match the stack: Adam, grad clip 5, hidden 32, 8
actions, `REWARD_SCALE=1e-3`.

---

## 6. Evaluation protocol (non-negotiable)

1. `python -m pytest -q`  
2. Pinned-seed H2H: `tournament_sub_vs_base.py --seed BASE --games N`  
3. Compare vs `baseline.zip` (Hybrid Direct), then vs Harvest Ledger / MuZero zips
4. Reject “improvements” that only win vs `baseline2` ($3k no-op)  
5. Mirror gate for refactors: identical seeds → identical action dicts  

---

## 7. Paper → file cheat sheet

| Paper idea | Do this in-repo |
| :--- | :--- |
| MuZero h/g/f + MCTS | `muzero.py` |
| Value-equivalent model | Prefer wealth features; hybrid `macro_mcts.py` |
| Reanalyse / Unplugged | `train_muzero_offline.py` Phase 2–3 + zip experts |
| EfficientZero SSL | Add consistency term on `(o_t,a,o_{t+1})` |
| Sampled MuZero | Only if action set grows past ~16 |
| Gumbel MuZero | Root selection under tight actTimeout |
| Stochastic MuZero | Chance nodes for prices / rival later |
| Predictron / VPN / TreeQN | Historical motivation for abstract planning — not required code |

---

## 8. Anti-patterns (learned the hard way)

1. **Planning over tile actions** — tree explodes; miss actTimeout → PASS/$3k.  
2. **Lazy-loading MuZero inside first `agent()`** — same timeout failure mode.  
3. **Mutating Hybrid/Harvest market every step from MuZero** — chassis state
   desync; scores collapse from ~$150k → ~$10–70k.  
4. **Trusting unpinned seeds** for ranking — seat+seed noise dominates.  
5. **Resurrecting CFR dual-game** — baseline beat those; keep them out of
   `step()`.  

---

## 9. Minimal “done” checklist

- [ ] `MuZeroExpansionPolicy` loaded with `muzero_checkpoints.pt`  
- [ ] Phase1–3 train script reproduces falling loss + smoke expand@day8→NE  
- [ ] Eager model load at import  
- [ ] Pinned 10-game H2H ≥ MacroMCTS / within striking distance of Hybrid money  
- [ ] Market ≤10 and shed emergency rules untouched  
- [ ] Optional: Gumbel root + consistency loss behind flags  

---

## 10. Suggested reading order (from this folder)

1. `papers/00_muzero_schrittwieser_2019.pdf` — core  
2. `papers/14_value_equivalence_grimm_2020.pdf` + `14b_…` — why latent models work  
3. `papers/03_muzero_unplugged_reanalyse_2021.pdf` — offline / reanalyse  
4. `papers/04_efficientzero_ye_2021.pdf` — SSL + sample efficiency  
5. `papers/09_gumbel_muzero_danihelka_2022.pdf` — few-sim search  
6. `notes/stochastic_muzero_summary.md` — stochastic markets later  
7. `papers/05_sampled_muzero_hubert_2021.pdf` — if actions grow  
8. `papers/11_lightzero_benchmark_2023.pdf` — engineering reference  

Then re-read `ARCHITECTURE_REVIEW.md` § Path to MuZero and `AGENTS.md` §4.4.
