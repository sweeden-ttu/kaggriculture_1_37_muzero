# End-to-End Data Flow

## Complete Pipeline: Observation → Action

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                           KAGGRICULTURE MUZERO DATA FLOW                                  │
└──────────────────────────────────────────────────────────────────────────────────────────┘

STEP 1: OBSERVATION ENCODING
════════════════════════════
Input: Raw Kaggle observation dict
       {
         "step": 123,
         "player": 0,
         "farms": [...],
         "market_prices": {...},
         "private": {"shed": {...}}
       }

Process: KaggricultureObservationEncoder.encode(obs)
         - Channel 0: Unlocked quadrant mask (NW, NE, SW, SE)
         - Channels 1-8: Tile entities (Empty, Weed, Crops×5, Structure)
         - Channels 9-13: Crop health (Watered, Unwatered days, Yield, Lifespan, Fertilized)
         - Channels 14-17: Livestock (Present, Fed, Unfed days, Care bonus)
         - Channels 18-19: Unit positions (Farmer, Hands)
         - Channels 20-27: Global context (Day, Hour, Log cash, Shed ratio, Prices)

Output: torch.Tensor [1, 28, 10, 10] (float32)

Code: muzero/observation_encoder.py:86-245
      muzero/observation_encoder.py:247-291 (get_action_mask)


STEP 2: INITIAL INFERENCE (ROOT NODE)
════════════════════════════════════
Input: obs_tensor [Batch, 28, 10, 10]

Process: KaggricultureMuZeroChassis.initial_inference(obs)
         1. h_θ (SpatialRepresentationNetwork):
            - Stem: Conv2d(28→64, 3×3, padding=1) + BN + ReLU
            - 3× ResNetBlock2D(64) with pre-activation
            - L2 normalize on channel dim → s_0 [Batch, 64, 10, 10]

         2. f_θ (SpatialPredictionNetwork):
            - Trunk: Conv2d(64→32, 1×1) + BN + ReLU → Flatten → Linear(3200→256) + ReLU
            - Farmer head: Linear(256→15)
            - Hands head:  Linear(256→32)
            - Market head: Linear(256→20)
            - Value head:  Linear(256→601)

Output: (s_0, preds_dict, value_scalar)
        s_0: [Batch, 64, 10, 10]
        preds_dict: {
            "logits_farmer": [Batch, 15],
            "logits_hands":  [Batch, 32],
            "logits_market": [Batch, 20],
            "value_logits":  [Batch, 601]
        }
        value_scalar: [Batch] (softmax(value_logits) @ support)

Code: muzero/chassis.py:288-292


STEP 3: MCTS ROOT EXPANSION
═══════════════════════════
Input: s_0, root_preds

Process: SampledMuZeroMCTS.search()
         1. Sample K=8 joint actions from factorized heads:
            joint_actions, joint_log_probs = prediction.sample_joint_actions(s_0, K=8)
            - Independent categorical sampling per head
            - Action masks applied (illegal → -1e9 logits)

         2. Apply Dirichlet noise at root:
            noise ~ Dirichlet(α=0.25 × K)
            priors = (1-ε) × softmax(log_probs) + ε × noise

         3. Create root children MCTSNode(prior, action_tuple)

Output: Root node with K=8 children, each with prior probability

Code: muzero/mcts.py:161-194


STEP 4: MCTS SELECTION (PUCT)
════════════════════════════
Loop: for _ in range(num_simulations=25):
        
        1. Selection: Traverse tree using PUCT score:
           Score(s,a) = Q_norm(s,a) + P(s,a) × √(ΣN(s,b)) / (1+N(s,a)) × (c1 + log((ΣN+c2+1)/c2))
           - MinMaxStats normalizes Q values across entire tree
           - c1=1.25, c2=19652

        2. Until leaf node (unexpanded) reached

Code: muzero/mcts.py:126-150


STEP 5: MCTS EXPANSION (RECURRENT INFERENCE)
══════════════════════════════════════════
Input: parent_node.latent_state [Batch, 64, 10, 10], action_tuple (f,h,m)

Process: 
         1. Convert action_tuple → action_plane [1, 16, 10, 10]:
            - Channels 0-4: Farmer one-hot (5 actions)
            - Channels 5-9: Hands one-hot (5 actions)  
            - Channels 10-15: Market one-hot (6 actions)

         2. g_θ (SpatialDynamicsNetwork):
            - Input: concat(s_prev, action_plane) [Batch, 80, 10, 10]
            - Stem: Conv2d(80→64) + BN + ReLU
            - 3× ResNetBlock2D(64)
            - L2 normalize → s_next [Batch, 64, 10, 10]
            - Reward head: Conv2d(64→16) → Flatten → Linear(1600→256) → Linear(256→601)

         3. f_θ (SpatialPredictionNetwork) on s_next:
            - Same as root but on unrolled state

Output: (s_next, reward_scalar, preds_dict, value_scalar)
        reward_scalar: softmax(reward_logits) @ support
        value_scalar:  softmax(value_logits) @ support

Code: muzero/chassis.py:294-303 (recurrent_inference)
      muzero/mcts.py:204-229 (expansion in search)


STEP 6: MCTS BACKUP
══════════════════
Process: Discounted value backpropagation:
         value = leaf_value
         for node in reversed(search_path):
             node.visit_count += 1
             node.value_sum += value
             value = node.reward + γ × value  (γ=0.997)
             min_max.update(value)

Output: Updated visit counts, value sums, MinMaxStats

Code: muzero/mcts.py:230-236


STEP 7: POLICY EXTRACTION
════════════════════════
Process: Visit-count policy π(a|s₀):
         total_visits = Σ child.visit_count
         π(a) = child.visit_count / total_visits
         best_action = argmax child.visit_count

Output: best_action (f,h,m), policy_dict, root_value

Code: muzero/mcts.py:238-245


STEP 8: ACTION TRANSLATION (CHASSIS PIPELINE)
════════════════════════════════════════════
Input: best_action tuple, current observation

Process: hybrid_chassis/pipeline.py:translate_macro_option()
         1. Map macro index → raw Kaggle actions
         2. Apply survival masks (action_masking.py)
         3. Execute through reactive layers:
            - hand_align, weed_repair, sell_lead, front_run
            - budget_guard, room_guard, clamp_sells
            - dead_stock, terminal_liquidation

Output: Kaggle action dict {"farmer": [...], "hands": [...], "market": [...]}

Code: hybrid_chassis/pipeline.py
      hybrid_chassis/layers/action_masking.py


STEP 9: TRAINING DATA COLLECTION
═══════════════════════════════
During self-play, each step stores:

GameTrajectory.append_step(
    obs: [28, 10, 10] numpy array,
    action: (f, h, m) tuple,
    reward: float (environment reward),
    policy_dict: {(f,h,m): visit_prob} from MCTS,
    value: root_value from MCTS,
    priority: 1.0 (initial)
)

Trajectory finalized with terminal observation.

Code: muzero/buffer.py:38-70 (GameTrajectory)
      muzero/buffer.py:73-213 (MuZeroTrajectoryBuffer)


STEP 10: REPLAY BUFFER SAMPLING (K-STEP UNROLL)
══════════════════════════════════════════════
Uniform sampling (MuZeroTrajectoryBuffer):
  1. Filter trajectories with len ≥ K
  2. Random trajectory, random start_idx
  3. Extract K+1 observations, K actions, K rewards
  4. Compute n-step bootstrap targets:
     z_t = Σ_{j=0}^{n-1} γ^j r_{t+j} + γ^n v_{t+n}
  5. Convert joint policy dicts → factorized targets:
     target_f[f] = Σ_{h,m} π(f,h,m)
     target_h[h] = Σ_{f,m} π(f,h,m)
     target_m[m] = Σ_{f,h} π(f,h,m)

PER sampling (PrioritizedMuZeroBuffer):
  1. priority = max_k |v_{t+k} - z_{t+k}| + ε
  2. P(i) ∝ priority^α (α=0.6)
  3. IS weights: w_i = (N × P(i))^{-β} / max(w)
  4. β annealed 0.4 → 1.0 over 100k frames

Output batch: {
    "obs": [Batch, K+1, 28, 10, 10],
    "actions": [Batch, K, 3],
    "rewards": [Batch, K],
    "values": [Batch, K+1],          # n-step targets
    "target_policy_farmer": [Batch, K+1, 15],
    "target_policy_hands":  [Batch, K+1, 32],
    "target_policy_market": [Batch, K+1, 20],
    "is_weights": [Batch, 1]         # PER only
}

Code: muzero/buffer.py:141-203 (uniform)
      muzero/buffer.py:254-329 (PER)


STEP 11: TRAINING STEP (K-STEP UNROLL LOSS)
═══════════════════════════════════════════
Input: batch from replay buffer

Process (PrioritizedMuZeroTrainer.train_step):
  
  1. Root pass (k=0):
     s_0, root_preds, root_val = online_model.initial_inference(obs[:,0])
     policy_loss = CE(root_farmer, target_f[:,0]) + CE(root_hands, target_h[:,0]) + CE(root_market, target_m[:,0])
     value_loss = MSE(root_val, target_values[:,0])  # per-sample for PER
     
  2. Recurrent unroll (k=1..K):
     for k in 1..K:
         action_plane = joint_action_to_plane(actions[:,k-1])
         s_k, r_pred, preds_k, v_pred = online_model.recurrent_inference(s_{k-1}, action_plane)
         
         policy_loss += CE(preds_k.farmer, target_f[:,k]) + CE(preds_k.hands, target_h[:,k]) + CE(preds_k.market, target_m[:,k])
         value_loss += MSE(v_pred, target_values[:,k])
         reward_loss += MSE(r_pred, rewards[:,k-1])
         
         # SimSiam Consistency
         target_z_k = target_model.representation(obs[:,k])  # No grad!
         c_loss = SimSiamConsistencyLoss(s_k, target_z_k, simsiam_head)
     
  3. Composite loss:
     L = mean_policy + 0.25×mean_value + mean_reward + λ_cons×mean_consistency
     (PER: weighted by IS weights per sample)
  
  4. Backward + optimizer step + EMA update:
     θ⁻ ← 0.995 × θ⁻ + 0.005 × θ

Code: muzero/trainer.py:93-184 (PER trainer)
      muzero/trainer.py:240-311 (uniform trainer)
      muzero/chassis.py:368-451 (compute_muzero_unroll_loss_with_consistency)


STEP 12: REANALYZE (TARGET REFRESH)
══════════════════════════════════
Periodic (every N steps):
  1. Sample trajectory from buffer
  2. For each timestep t:
     obs_t = trajectory.observations[t]
     best_act, fresh_policy, fresh_val = mcts_engine.search(obs_t, add_dirichlet_noise=False)
     trajectory.policies[t] = fresh_policy
     trajectory.values[t] = fresh_val
  3. Priorities reset (will be recalculated on next sample)

Code: muzero/buffer.py:205-212 (reanalyze_trajectory)
      muzero/trainer.py:186-194 (run_reanalyze_pass)
      muzero/trainer.py:313-320 (uniform trainer)


STEP 13: CONTINUOUS LOOP ORCHESTRATION
══════════════════════════════════════
pipelines/self_improving_loop.py:
  - SelfPlayWorker: generates trajectories using current champion
  - Trainer: consumes trajectories, updates model
  - Evaluator: runs tournaments vs baselines
  - Promoter: promotes champion if eval threshold met
  - ReplayManager: enforces disk budget (10GB/24GB), prunes oldest selfplay_*.json

train.sh (standard): LoRA rank=4, consistency 0.1→0.5
boost.sh (inverted): LoRA rank=16, consistency 1.0→0.5, antagonistic gradients

Code: pipelines/self_improving_loop.py
      pipelines/replay_manager.py
      train.sh / boost.sh
```

## Tensor Shape Summary

| Stage | Tensor | Shape |
|-------|--------|-------|
| Observation | Raw dict | - |
| Encoded | Spatial tensor | [B, 28, 10, 10] |
| h_θ output | Latent state s₀ | [B, 64, 10, 10] |
| Action plane | One-hot action | [B, 16, 10, 10] |
| g_θ output | Next latent sₖ | [B, 64, 10, 10] |
| g_θ reward | Reward logits | [B, 601] |
| f_θ policy | Farmer logits | [B, 15] |
| f_θ policy | Hands logits | [B, 32] |
| f_θ policy | Market logits | [B, 20] |
| f_θ value | Value logits | [B, 601] |
| MCTS policy | Visit counts | Dict[(f,h,m), prob] |
| Batch obs | K-step slice | [B, K+1, 28, 10, 10] |
| Batch actions | K-step actions | [B, K, 3] |
| Batch targets | Factorized policy | [B, K+1, 15/32/20] |
| Batch values | n-step bootstrap | [B, K+1] |

## Key Constants (from spatial_constants.py)

```python
OBS_CHANNELS = 28
LATENT_CHANNELS = 64
ACTION_CHANNELS = 16
NUM_FARMER_ACTIONS = 15
NUM_HAND_ASSIGNMENTS = 32
NUM_MARKET_ORDERS = 20
SUPPORT_SIZE = 601
SUPPORT_B = 300
DISCOUNT = 0.997
```