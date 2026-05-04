# MEGA Benchmark

Benchmark results of MEGA tech report:

1. **SkillsBench skill-curation comparison** (84 tasks, 11 domains) — MEGA (WG) vs No Skills / SkillNet / AgentSkillOS. Results in [`skillbench-results/`](skillbench-results/).
2. **GEPA-style workflow benchmarks** (HotpotQA, IFBench, HoVer, PUPA) — comparison against MIPROv2, TextGrad, GEPA, Feedback Descent. Results in [`optimization-results/`](optimization-results/).

---

## 1. SkillsBench skill-curation comparison (Gemini 3 Flash, 84 tasks)

Same agent, same Docker tasks, same 4,207-asset skill pool — only the skill discovery / orchestration method varies. Pass rate is the primary metric; efficiency = pass rate per megatoken.

| Condition | Pass Rate (%) | Avg Tokens/Task (k) | Curation Latency (s/task) | Efficiency (score/Mtok) |
|---|:-:|:-:|:-:|:-:|
| No Skills | 31.5 | 894 | — | 0.353 |
| AgentSkillOS | 41.1 | 1189 | 403.4 | 0.345 |
| SkillNet | 41.7 | 983 | 37.8 | 0.424 |
| **MEGA (WG)** | **46.5** | **822** | **11.8** | **0.566** |

MEGA achieves the highest pass rate **with the lowest tokens** and **fastest curation** simultaneously — efficiency 0.566 score/Mtok = 1.33× SkillNet, 1.64× AgentSkillOS. Trial bundles published as the [`v0.1.0-jobs`](https://github.com/mega-edo/mega_benchmark/releases/tag/v0.1.0-jobs) GitHub Release.

## 2. GEPA workflow benchmarks (GPT-4.1 Mini)

All methods evaluated on `gpt-4.1-mini-2025-04-14` over the same train/test splits and seeds. Baseline / MIPROv2 / TextGrad / GEPA scores from GEPA paper [Agrawal et al. 2026]; Feedback Descent from [Lee et al. 2025]. MEGA optimizes the same GEPA-faithful agent (DSPy/GEPA scaffolding stripped) at the **code + prompt** level.

| Method | HotpotQA | IFBench | HoVer | PUPA | **Agg.** |
|---|---:|---:|---:|---:|---:|
| Baseline | 38.00 | 47.79 | 46.33 | 78.57 | 52.67 |
| MIPROv2 | 58.00 | 49.15 | 48.33 | 83.37 | 59.71 |
| TextGrad | 62.33 | 48.64 | 47.67 | 85.68 | 61.08 |
| Feedback Descent | 68.33 | 54.59 | 57.67 | 85.66 | 66.56 |
| GEPA (best) | 69.00 | 55.95 | 56.67 | 96.46 | 69.52 |
| **MEGA** | **72.67** | **61.05** | **74.67** | **97.81** | **76.55** |

MEGA aggregate **+7.03 over GEPA**, **+9.99 over Feedback Descent**. Per-task raw outputs in [`optimization-results/<task>/results/<task>_optimized.json`](optimization-results/).

Splits used (train/val/test): HotpotQA 150/100/300, IFBench 150/100/294, HoVer 150/100/300, PUPA 111/111/221. MEGA uses smaller val sets than GEPA's original splits to reduce optimization cost.
