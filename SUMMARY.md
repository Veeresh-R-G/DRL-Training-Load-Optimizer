
# DRL TRAINING LOAD OPTIMIZER — PROJECT SUMMARY


### Architecture
- Agent         : PPO (Proximal Policy Optimization)
- State encoder : Transformer (2 layers, 4 heads, d_model=64)
                  Mean pooling over 8-week history window
- Action space  : [intensity ∈ [0.75,1.4], volume_km, workout_mix]
- Training      : Curriculum learning (4-week → 8-week → 12-week)
- Total steps   : 600,000 (150k + 200k + 250k)
- Best reward   : +19.25

### Key Design Decisions (beyond the paper)

1. ACWR as ONE of 4 injury signals, not sole predictor
   → Impellizzeri et al. (2020): ACWR alone is insufficient
2. Compliance-gated fitness reward
   → CTL only credited when achieved through doable plans
3. Phase-aware TSB targets
   → Different goals during build vs taper phases
4. Curriculum RL for long-horizon credit assignment
   → 4-week plan first, then 8-week, then 12-week

### Results vs Static Baseline (intermediate athlete, seed=42)
| Metric            | Static Plan | DRL Agent | Note                                     |
|-------------------|-------------|------------|------------------------------------------|
| Final CTL         | 58.4        | 41.8       | Static optimises single objective        |
| Final TSB         | +2.0        | -3.5       | Target: +5 to +25                        |
| Total Injuries    | 0.0         | 3.0        |                                          |
| Avg Compliance    | 84.4%       | 62.6%      |                                          |

### Monte Carlo (50 runs)
| Metric          | Static Plan | DRL Agent |
|-----------------|-------------|------------|
| Avg Injuries    | 0.92        | 1.30       |
| Avg Compliance  | 78.1%       | 75.4%      |
| Avg CTL         | 58.7        | 42.2       |

### Interpretation
The CTL gap reflects a deliberate multi-objective tradeoff.
Static plans maximise fitness by ignoring athlete response.
The DRL agent sacrifices raw CTL to maintain compliance
and injury safety — Runna's core product value proposition.


#### References : 
1. Xia, X., Chen, Q. & Wang, Z. (2025). *Deep reinforcement learning-driven 
   personalized training load control algorithm for competitive sports performance 
   optimization.* Scientific Reports, 16, 845.  
   [Full text (open access)](https://doi.org/10.1038/s41598-025-30453-z)    
    
2. Links to reference
[Nature official](https://www.nature.com/articles/s41598-025-30453-z) | 
[PubMed](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12779991/)
