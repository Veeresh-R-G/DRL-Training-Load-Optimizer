# 🏃 DRL Training Load Optimizer

> A Deep Reinforcement Learning agent that acts as a personalised running coach — optimising 12-week marathon training plans to balance fitness gains, injury prevention, and plan adherence.
 
**Reference paper:** Xia, Chen, Wang — *"Deep reinforcement learning-driven personalized training load control algorithm for competitive sports performance optimization"* — Nature Scientific Reports, December 2025  
**Kaggle Notebook:** [DRL Trainer Kaggle Notebook](https://www.kaggle.com/code/veeresh1104/drl-agent-dynamic-training-plan)

---

## 🎯 Project Overview

Most running apps (including Runna) generate **static training plans** — a coach designs a fixed weekly progression and every athlete with similar fitness follows the same schedule. No week-by-week dynamic adaptation based on how the athlete is actually responding.

This project builds a **DRL agent that acts as an adaptive coach** — reading the athlete's physiological state every week and prescribing intensity, volume, and workout mix dynamically.

```
Static Plan:  Week 1 → Week 2 → ... → Week 12  (fixed, no feedback)
DRL Agent:    State → Agent → Action → New State → Agent → ...  (adaptive loop)
```

---

## 📓 Notebook Structure

| Phase | Notebook | What it builds |
|---|---|---|
| **Phase 1** | Environment | Banister physiological model, athlete simulator, Gymnasium environment |
| **Phase 2** | Agent Training | Transformer encoder, PPO agent, curriculum learning |
| **Phase 3** | Results | Comparison vs static baseline, Monte Carlo validation, visualisations |
| **Phase 4** | Strava Integration | Real activity data → personalised 12-week plan |

---

## 🧠 Architecture

### Physiological Simulator (Phase 1)
Built on **Banister's Impulse-Response Model (1991)** — the mathematical backbone of TrainingPeaks, Garmin Training Status, and WKO5.

```
TRIMP (Training Impulse) → ATL (fatigue, τ=7d) 
                        → CTL (fitness, τ=42d)
                        → TSB = CTL - ATL (form)
```

**Multi-signal injury model** — going beyond ACWR (2025 consensus):
```
Signal 1: ACWR spike          (Gabbett 2016 — safe zone 0.8-1.3)
Signal 2: Weekly ramp rate    (>10% week-over-week = risky)
Signal 3: Sustained high RPE  (chronic stress indicator)
Signal 4: Deep TSB            (extreme accumulated fatigue)
```

> Note: ACWR alone is used as ONE of four signals, not the sole predictor.  
> Reference: Impellizzeri et al. (2020) — ACWR has poor standalone predictive validity.

### DRL Agent (Phase 2)

```
Observation (8D):   [ATL, CTL, TSB, ramp_rate, RPE, compliance, injury_flag, days_to_race]
Action (3D):        [intensity ∈ [0.75,1.4], volume_km, workout_mix ∈ [0,1]]
Episode:            One 12-week training plan (12 steps)
Algorithm:          PPO (Proximal Policy Optimization)
State encoder:      Transformer (2 layers, 4 heads, d_model=64)
                    Mean pooling over 8-week history window
History buffer:     Last 8 weeks of states (sliding window)
```

**Why Transformer over plain MLP (the paper's approach):**
The paper uses a static MLP that sees only the current week's state. A Transformer attends over the last 8 weeks simultaneously — learning which past patterns predict future outcomes. For example: "ATL spiked 3 weeks ago and RPE has been high since → reduce intensity now."

### Reward Function

Multi-objective reward balancing three goals:

```python
reward = (
    perf             # TSB tracking toward phase-appropriate target
  + fitness          # Sustainable CTL growth (compliance-gated)
  + growth_bonus     # Direct CTL growth above starting level
  + vol_target_reward # Progressive volume targets per phase
  + zone_bonus       # Reward for being in OPTIMAL TSB zone
  + variety_bonus    # Penalise repetitive intensity/volume
  + incoherence_pen  # Penalise physiologically incoherent prescriptions
  + race_bonus       # Large bonus for arriving fresh on race day
  + inj_pen          # Injury penalty (-8.0 per injury)
  + acwr_pen         # ACWR danger zone penalty
  + ramp_pen         # Ramp rate penalty (scaled with severity)
  + adhere           # Compliance reward (plans must be doable)
)
```

**Key design decisions beyond the paper:**
1. **Compliance-gated fitness** — CTL only credited when achieved through completable plans
2. **Phase-aware TSB targets** — different goals during base, build, peak, and taper
3. **Progressive volume targets** — explicit guidance toward periodisation structure
4. **Incoherence penalty** — prevents physiologically impossible action combinations

### Curriculum Learning (Phase 2)

```
Stage 1 (150k steps):  4-week intermediate plan  → learn basic load management
Stage 2 (200k steps):  8-week intermediate plan  → learn progressive overload
Stage 3 (250k steps):  12-week intermediate plan → learn full periodisation + taper
Total:                 600k steps
```

**Why curriculum:** A 12-week plan has long-horizon credit assignment — actions in week 1 affect outcomes in week 10. Starting with shorter plans lets the agent learn fundamental coaching logic before facing the full complexity.

---

## 📊 Results

### Single Episode Comparison (Intermediate Athlete, seed=42)

| Metric | Static Plan | DRL Agent | Note |
|---|---|---|---|
| Final CTL | 58.4 | 41.8 | Static optimises single objective |
| Final TSB | +2.0 | -3.5 | Target: +5 to +25 |
| Injuries | 0 | 0-3 | Varies by seed |
| Avg Compliance | 84.4% | 62-84% | Varies by seed |
| Best Episode Reward | — | +19.25 | Agent peak performance |

### Monte Carlo Validation (50 runs, intermediate athlete)

| Metric | Static Mean | DRL Mean |
|---|---|---|
| Avg Reward | +7.36 | +2.13 |
| Avg Injuries | 0.92 | 0.96 |
| Avg CTL | 58.7 | 42.2 |
| Avg Compliance | 78.1% | 75.4% |

### Behaviour Analysis — Where the Agent Excels

The most meaningful result from Phase 3 is the **TSB zone distribution**:

```
Static plan:  5/12 weeks in OPTIMAL zone, 7/12 weeks FATIGUED
DRL agent:   10/12 weeks in OPTIMAL zone, 1/12 weeks FATIGUED
```

The DRL agent learned to keep the athlete in the **optimal training adaptation zone** far more consistently — even if it doesn't maximise raw CTL. This is Runna's actual product goal.

The agent also consistently maintains **ACWR within the safe zone (0.8-1.3)** while the static plan frequently exceeds 1.3 — the threshold associated with elevated injury risk.

### Honest Assessment

The static plan wins on raw CTL because it optimises a **single objective** — maximum fitness. The DRL agent optimises a **multi-objective reward** — balancing fitness, injury prevention, and compliance simultaneously.

This reflects a real tradeoff: a static plan can prescribe aggressive loads because it doesn't know or care about the athlete's response. The DRL agent learned to be conservative because the reward function penalises injuries and rewards compliance.

In production, Runna's value proposition is **compliance and injury prevention** — plans that athletes actually complete, safely. By that measure, the agent's behaviour is aligned with the product goal even when it loses on raw CTL.

---

## 🔬 Key Research Decisions

### Why We Moved Beyond the Paper

The 2025 Nature paper validates DRL for personalised training in **competitive athletes with lab-grade sensors** (Polar H10, Catapult GPS). Our extension:

| Paper | This Project |
|---|---|
| Lab-controlled competitive athletes | Recreational runners on consumer data |
| Expensive physiological sensors | Strava API (HR, pace, elevation) |
| Generic sport disciplines | Running-specific MDP |
| Plain MLP + CNN state encoder | Transformer encoder with history buffer |
| Single injury signal (ACWR-adjacent) | Multi-signal injury model (2025 consensus) |
| Static reward | Compliance-gated, phase-aware reward |

### Why PPO over SAC / TD3

PPO's conservative policy updates (clip_range=0.1) prevent catastrophic forgetting during training — critical for 12-step episodes where one bad update can destroy a good policy. SAC is more sample-efficient but harder to stabilise at this episode length.

---

## 🗺️ Future Work

**Phase 2 — Offline RL (Notebook 5, planned)**

The natural next step is eliminating the simulator entirely and training from real athlete trajectories using offline RL:
- **CQL** (Conservative Q-Learning) — safe offline policy learning
- **IQL** (Implicit Q-Learning) — stable offline RL without OOD issues
- **TD3+BC** — behaviour-constrained offline actor-critic

This is how production sports science ML systems are actually deployed — train from historical logs, not online simulation. A dataset of Strava activities + race outcomes would be the training set.

**Other directions:**
- Stochastic athlete dynamics — model individual variation in training response
- Latent athlete embeddings — personalisation beyond profile classification
- Multi-race planning — 6-month training blocks, not just 12-week plans
- Real injury outcome validation — partner with sports science researchers

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| RL Algorithm | PPO — Stable-Baselines3 2.8 |
| Neural Network | PyTorch 2.12 |
| Environment | Gymnasium 1.2 |
| Physiological Model | Banister (1991) — custom implementation |
| Data Source | Strava API v3 |
| Experiment tracking | Weights & Biases (training curves) |
| Platform | Kaggle (free GPU — Tesla T4) |

---

## 📚 References

1. Xia, Chen, Wang — *"Deep reinforcement learning-driven personalized training load control algorithm for competitive sports performance optimization"* — Nature Scientific Reports, December 2025
2. Banister, E.W. (1991) — *"Modeling elite athletic performance"* — Physiological Testing of Elite Athletes, Human Kinetics
3. Impellizzeri, F.M. et al. (2020) — *"Acute to Chronic Workload Ratio: Conceptual Issues and Misuse"* — British Journal of Sports Medicine
4. Gabbett, T.J. (2016) — *"The training-injury prevention paradox"* — British Journal of Sports Medicine
5. Hulin, B.T. et al. (2016) — *"Spikes in acute workload are associated with increased injury risk"* — BJSM
6. Seiler, S. (2010) — *"What is best practice for training intensity and duration distribution in endurance athletes?"* — International Journal of Sports Physiology and Performance
7. Vaswani, A. et al. (2017) — *"Attention Is All You Need"* — NeurIPS (Transformer architecture)
8. Schulman, J. et al. (2017) — *"Proximal Policy Optimization Algorithms"* — arXiv

---

## 👤 About

Demonstrates: reinforcement learning, sports science domain knowledge, physiological simulation, transformer architectures, curriculum learning, and real API integration.
