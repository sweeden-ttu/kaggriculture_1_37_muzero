# League Tracker Pfsp - Architecture Specification

**Source**: evaluation/league.py, league_tracker.py
**Documentation**: docs/League_Tracker.md
**Generated**: 2026-09-27T03:38:05.486423

## Description
AlphaStar-style league with PFSP matchmaking: hard/var/uniform modes, role-specific ratios

## Key Classes

## Key Functions

## Hyperparameters

## Architecture Notes
- LLM analysis failed: [Errno 61] Connection refused

## Training Workflow
- LeagueTracker maintains checkpoints list + wins_matrix + games_matrix
- Laplace smoothing: win_rate = (wins + 1) / (games + 2)
- PFSP modes: hard=(1-x)^p, var=x(1-x), uniform=1
- Main agent: 35% self-play, 50% hard PFSP, 15% var PFSP
- Main exploiter: 50% vs current main, 50% var PFSP past mains
- League exploiter: 100% hard PFSP (p=3.0)

## Testing Approach
- Unit test: add_checkpoint() expands matrices correctly
- Test get_win_rate() with Laplace smoothing: 0 games → 0.5
- Test PFSP hard mode: low win_rate → high weight
- Test PFSP var mode: win_rate≈0.5 → highest weight
- Test select_matchmaking_opponent() respects role ratios
- Integration: tournament.py uses league for opponent selection

## Evaluation Metrics
- League diversity: win-rate matrix heatmap
- PFSP effectiveness: main agent win-rate vs hard/var/uniform opponents
- Exploiter performance: main_exploiter win-rate vs current main
- Population collapse detection: exploiters >70% win-rate trigger reset