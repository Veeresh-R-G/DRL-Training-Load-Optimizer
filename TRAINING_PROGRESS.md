# 🔬 Training Progress Log — DRL Running Coach

> A full history of every experiment, what was tried, what worked, and what didn't.  
> Kept for personal reference and to demonstrate research depth to interviewers.

---

## The Core Problem We Were Solving

Train a PPO agent to prescribe 12-week marathon training plans that:
1. Build fitness (CTL) progressively
2. Avoid injury
3. Produce plans athletes actually complete (compliance)

The challenge: all three objectives conflict. The agent kept finding corners of the reward landscape where it could score well on one metric while ignoring the others.

---

## Run History

---

### Run 1 — Baseline (100k steps)
**Config:**
- PPO default hyperparams: clip=0.2, LR=3e-4, batch=64, epochs=10
- Fitness weight: 1.0
- Injury penalty: -5.0
- Compliance weight: 0.5

**Results:**
```
Reward: -2.44 → -7.52  (collapsed)
CTL:    78.8            (very high)
Injuries: 0.20
Compliance: 77%
```

**What happened:** Agent overtrained aggressively. Found that spiking volume early builds CTL fast. Reward looked good initially then collapsed as injuries and poor compliance accumulated. First example of **policy collapse** — the agent found a local optimum then fell off it.

**Lesson:** Fitness weight of 1.0 was too strong. Agent prioritised CTL above everything.

---

### Run 2 — Reduce Fitness Weight (500k steps)
**Changes:**
- Fitness weight: 1.0 → 0.6
- Injury penalty: -5.0 → -8.0
- Compliance weight: 0.5 → 1.0

**Results:**
```
Reward: +4.19 → -4.60  (still collapsed)
CTL:    78.9            (same as Run 1)
Injuries: 0.40
Compliance: 75%
```

**What happened:** Changing the fitness weight made almost no difference to CTL — the agent still reached 78. This revealed that CTL growth wasn't primarily driven by the fitness reward term. The agent found that high volume alone builds CTL regardless of the weight. Changing the multiplier from 1.0 to 0.6 didn't change the exploitation strategy.

**Lesson:** The simulator itself was exploitable. High volume at any intensity drives CTL up. We needed to make the simulator consequences of overtraining more severe.

---

### Run 3 — Sustainable CTL + Tighter Simulator (500k steps)
**Changes:**
- Compliance-gated fitness: `sustainable_ctl = ctl * compliance`
- Ramp penalty scaled with severity (not binary)
- ACWR penalty scaled with severity
- TSB zone bonus added
- Injury recovery extended: 1-3 weeks → 2-5 weeks
- Effective volume: `actual_volume * (0.5 + 0.5 * compliance)`

**Results:**
```
Reward: +5.55 → -6.51  (collapsed again)
CTL:    67.5
Injuries: 0.65
Compliance: 75%
```

**What happened:** Compliance-gating helped but didn't stop the collapse. The agent still found ways to exploit the simulator — training hard until compliance dropped, accepting the penalty, then recovering. Injury recovery now 2-5 weeks made things more severe but agent adapted its exploitation pattern.

**Lesson:** Policy collapse was a training stability issue, not just a reward issue. We needed tighter PPO hyperparameters.

---

### Run 4 — PPO Stabilisation + Early Stopping (500k steps)
**Changes:**
- LR: 3e-4 → 1e-4 (decaying to 1e-5)
- n_steps: 512 → 1024
- batch_size: 64 → 256
- n_epochs: 10 → 5
- clip_range: 0.2 → 0.1
- ent_coef: 0.01 → 0.005
- Added EvalCallback with StopTrainingOnNoModelImprovement

**Results:**
```
Reward: +0.95 → +3.17  (STABLE — no collapse!)
CTL:    19.4            (barely moved)
Injuries: 0.50
Compliance: 82%
```

**What happened:** Policy collapse fixed. Reward was stable throughout training. But new problem emerged — agent was too conservative. CTL barely moved from starting value (20). Agent found a new local optimum: prescribe easy plans, avoid all risk, collect small positive rewards. The tighter clip range prevented learning as well as preventing collapse.

**Lesson:** Stability and learning are in tension. Fixed one, broke the other.

---

### Run 5 — Curriculum Learning (600k steps — first attempt)
**Changes:**
- Added curriculum: 4-week beginner → 8-week intermediate → 12-week intermediate
- Stage 1: 150k steps, Stage 2: 200k steps, Stage 3: 250k steps
- Restored slightly higher exploration: ent_coef=0.02 for Stage 1
- No early stopping in Stage 3

**Results:**
```
Stage 1: +1.73  Stage 2: -1.01  Stage 3 final: +3.67
Best checkpoint: +13.47
CTL: 55.5  Injuries: 0.10  Compliance: 82%
```

**What happened:** Best run so far. Curriculum helped the agent learn progressively. Stage 2 dip (negative transfer) is expected when task difficulty increases. Stage 3 recovered and the best checkpoint hit +13.47. CTL reached 55 — first time we saw real fitness building. Injuries dropped to 0.10 — excellent.

**Lesson:** Curriculum learning works. The 4-week plan gives the agent a simpler version of the problem to learn first, then it transfers that knowledge to longer plans.

---

### Run 6 — Cell 11 Inspection — Intensity Exploit Discovered
**What happened (no training):**  
Ran Cell 11 (plan inspector) and discovered the agent was prescribing:
```
Every week: intensity=0.60 (minimum), volume varies, dist=1.0 (maximum)
```
Agent found that minimum intensity + maximum workout distribution = incoherent but safe. Low TRIMP from low intensity meant no injury risk, but high distribution score tricked the variety bonus.

**Changes made:**
- Raised intensity floor: ACTION_LOW 0.60 → 0.75
- Added incoherence penalty for high distribution + low intensity
- Added intensity variety bonus
- Added volume variety bonus
- Added race week TSB bonus (+3.0 for arriving fresh)
- Added CTL growth bonus above initial_ctl
- Added progressive volume targets per training phase

---

### Run 7 — New Reward Function (600k steps)
**Results:**
```
Stage 1: -4.70  Stage 2: +4.52  Stage 3: +11.60
Best checkpoint: +16.83
CTL: 41  Injuries: 0.15  Compliance: 84%
```

**What happened:** Best training metrics yet. Stage 3 reward strong and stable. Best checkpoint +16.83. But Cell 11 inspection showed agent still stuck at intensity=0.75 (new minimum) with dist=0.10 (new minimum). Agent shifted corners — now exploiting both lower bounds simultaneously instead of one.

**Lesson:** Raising floors doesn't stop exploitation — it just changes which floor the agent sits on. The problem is the reward function makes "do minimum" a viable strategy.

---

### Run 8 — Volume Minimum + Intensity Variety (600k steps)
**Changes:**
- min_weekly_km: 15 → 35 for intermediate
- Added _volume_history tracking
- Enhanced variety bonus: penalises both intensity AND volume repetition
- Added _damage_accumulator (delayed injury accumulation)
- Added stochastic noise to simulator (noise_scale=0.1)

**Results:**
```
Stage 1: -11.24  Stage 2: +5.92
CTL: 41  (flat)  Injuries: 0.50  Compliance: 80%
```

**What happened:** Agent immediately found new corner: volume=35 (new minimum), intensity=0.75 (minimum), every week without variation. The stochastic noise made things worse — it broke the static plan comparison baseline. Static plan started getting 10+ injuries per run because the noise made sustained high-intensity training catastrophically risky.

**Lesson:** Stochastic noise is valuable in theory (forces robust policies) but broke our comparison baseline. Need to separate training noise from evaluation noise, or not use noise at all at this stage.

---

### Run 9 — Revert Noise, Add Intensity Target (600k steps)
**Changes:**
- Removed stochastic noise entirely
- Added progressive intensity targets per plan phase
- Stage 3 extended to 250k steps
- Stage 1 changed from beginner to intermediate profile

**Results:**
```
Stage 1: -0.89  Stage 2: +0.98  Stage 3: +5.37
Best checkpoint: +10.98
CTL: 38  Injuries: 3 (beginner)  Compliance: 56%
```

**What happened:** Worse than Run 7. The intensity target reward conflicted with the injury penalty — agent trying to hit intensity targets got penalised for the resulting ACWR spikes. Stage 3 results weaker. Intensity target reward and damage accumulator together made the environment too punishing.

---

### Run 10 — Clean Revert to Best Config (600k steps)
**Changes:**
- Removed intensity target reward
- Removed damage accumulator
- Kept: vol_target_reward, growth_bonus, variety bonuses, incoherence_pen, race_bonus
- Stage 1 back to intermediate 4-week
- Cell 20 uncommented (best model loads correctly)
- Static plan intensity profile reverted to original

**Results:**
```
Stage 1: -3.72  Stage 2: -2.18  Stage 3: +4.77
Best checkpoint: +19.25
CTL: 41  Injuries: 0.96 (MC)  Compliance: 75%
```

**Final comparison (intermediate athlete):**
```
Metric          Static    DRL
Final CTL         58.4   41.8
Final TSB         +2.0   -3.5
Injuries           0.0    0-3
Compliance        84.4%  62-84%
TSB zone OPTIMAL  5/12   10/12  ← agent wins this
ACWR in safe zone rarely always  ← agent wins this
```

---

## Summary of What We Learned

### What Works
- ✅ Curriculum learning — essential for long-horizon credit assignment
- ✅ PPO clip_range=0.1 with decaying LR — prevents policy collapse
- ✅ Compliance-gated fitness reward — stops pure CTL maximisation
- ✅ Phase-aware TSB targets — guides agent through periodisation structure
- ✅ Multi-signal injury model — more physiologically accurate than ACWR alone
- ✅ Progressive volume targets — gives agent explicit periodisation guidance
- ✅ Transformer state encoder — learns which historical weeks matter most

### What Didn't Work
- ❌ Stochastic noise during training — broke comparison baseline
- ❌ Intensity target reward — conflicted with injury penalty
- ❌ Damage accumulator alone — too punishing when combined with other penalties
- ❌ Raising action space floors — agent just finds new corners
- ❌ Very high fitness weight (1.0) — causes overtraining and policy collapse
- ❌ Early stopping (StopTrainingOnNoModelImprovement) — triggers too early on noisy rewards

### The Persistent Challenge

The agent consistently found **local optima at action space boundaries**:
- Run 1-3: max volume, variable intensity → CTL 78 but injuries/collapse
- Run 4-5: min everything → CTL 20, stable but useless
- Run 6-7: intensity=0.75, dist=1.0 → incoherent but safe
- Run 8+: volume=35 (min), intensity=0.75 (min) → safe but not training

The root cause: in our simulator, the safest possible action (minimum volume, minimum intensity) produces small but consistently positive rewards. Building fitness requires taking risks (higher volume, higher intensity) which the simulator can punish unpredictably. A risk-averse agent rationally prefers the safe corner.

**This is a fundamental limitation of training RL in simulators.** The simulator's reward landscape doesn't perfectly capture real running coaching tradeoffs. A production system would use offline RL on real athlete historical data — bypassing the simulator entirely.

### The Honest Result

The agent learned meaningful physiological patterns:
- Keeps ACWR within safe zone (static plan frequently exceeds 1.3)
- Spends 10/12 weeks in OPTIMAL TSB zone (static: only 5/12)
- Maintains consistent compliance without catastrophic injury episodes

But it didn't learn to build CTL progressively — the static plan wins on raw fitness.

**The right framing:** DRL optimises for athlete experience (safe, consistent, doable) while static plans optimise for maximum fitness. Both are valid objectives. Runna's product leans toward the former.

---

## Recommended Next Steps (Future Work)

1. **Offline RL (CQL/IQL)** — train from real Strava activity logs instead of simulator
2. **Stochastic simulator with separate eval noise** — train robust, evaluate fair
3. **Longer training** (2-5M steps) — curriculum RL needs more time than we gave it
4. **Hyperparameter sweep with W&B** — systematic search instead of manual tuning
5. **Real user study** — recruit 10 runners, compare DRL vs static plan outcomes over 8 weeks

---

## Timeline

| Date | Milestone |
|---|---|
| Session 1 | Project scoped, Strava/Runna research, paper identified |
| Session 2 | Phase 1 complete — Banister model, simulator, Gym env, 25 unit tests |
| Session 3 | Phase 2 start — Transformer encoder, PPO agent, Runs 1-4 |
| Session 4 | Curriculum learning implemented, Runs 5-7 |
| Session 5 | Exploit discovery, reward engineering, Runs 8-10 |
| Session 6 | Phase 3 complete — comparison, Monte Carlo, visualisations |
| Session 7 | Phase 4 complete — Strava API, personal plan generation |
| Session 8 | README and progress log written |
