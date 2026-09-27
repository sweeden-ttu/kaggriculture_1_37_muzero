# Stochastic MuZero (ICLR 2022) — summary

**Paper:** Planning in Stochastic Environments with a Learned Model  
**Authors:** Antonoglou, Schrittwieser, Ozair, Hubert, Silver  
**OpenReview:** https://openreview.net/forum?id=X6D9bAHhBQ1  
**PDF:** not auto-downloaded (OpenReview 403); fetch manually if needed.

## Idea
Factor transitions into:
1. Deterministic action-conditioned move to an **afterstate** \(as_t\)
2. Stochastic chance transition from afterstate → next state \(s_{t+1}\) with outcome \(c\)

## Search
Alternating **decision nodes** (P-UCT over actions) and **chance nodes** (sample \(c \sim Pr(c|as)\)).

## Model extras vs MuZero
- Afterstate dynamics \(\phi\)
- Afterstate prediction \(\psi\)
- Chance outcomes via VQ-VAE-style one-hot codebook; chance loss + commitment cost

## Results (from author blog)
- Matches MuZero on deterministic Go
- Beats MuZero on 2048 (learned stochasticity; no hand simulator)
- Strong Backgammon, scales with simulations; ~21 chance outcomes ≈ dice rolls

Source blog archived: `stochastic_muzero_julian_blog.html`
