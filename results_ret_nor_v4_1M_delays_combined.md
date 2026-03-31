# VDPO $Ret_{nor}$ — MuJoCo v4, 1M steps (all delay grids combined)

## Setup

- **Metric**: $Ret_{nor} = (Ret_{alg} - Ret_{rand}) / (Ret_{df} - Ret_{rand})$ (same as the paper Table 2 definition; baselines: `random`, `delay0_sac`).
- **Target global step**: 1,000,000.
- **Delays 4 / 8 / 16** — VDPO logs: `output/slurm_logs/g1-pickup-grid-search/411547` (job `411547`).
- **Delays 5 / 25 / 50** — VDPO logs: `output/slurm_logs/g1-pickup-grid-search/411035` (job `411035`).
- **Baselines**: `output/slurm_logs/vdpo-baselines/411339`.
- **Seeds**: 0, 1, 2 (cells show mean±std and `(n/3)` when partial).

## Incomplete VDPO tasks (no line at target global step)

- Short-delay job `411547`: `[15]`.
- Mid-delay job `411035`: `none`.

## Table 1: $Ret_{nor}$ (action-delay columns) + baseline raw returns (last two columns)

Delay columns **4–50** are **normalized** $Ret_{nor}$. The last two columns are **raw** evaluation returns (`ep_re` at the target global step) for the `random` policy and for **delay-free SAC** (`delay0_sac`).

### Experimental conditions (for screenshots / collaborators)

| Item | Value |
| --- | --- |
| **MuJoCo / task API version** | **Gymnasium MuJoCo v4** (`HalfCheetah-v4`, `Hopper-v4`, `Ant-v4`, `Walker2d-v4`) |
| **Training horizon** | **1,000,000** global environment steps per run (**1M steps** = **1 million** steps total per run) |
| **Metric reporting point** | $Ret_{alg}$, $Ret_{rand}$, $Ret_{df}$ all taken at the same **global step = 1,000,000** (aligned with paper Table 2: 1M-step reporting) |
| **Random seeds** | **3** runs per cell when complete: seeds **0, 1, 2** |
| **Action delays in table** | **4, 8, 16** (job `411547`) and **5, 25, 50** (job `411035`) |

*MuJoCo **v4** tasks; each training run uses **one million** environment steps.*

| Task | Method | Action Delay | 4 | 8 | 16 | 5 | 25 | 50 | random (raw ep_re) | delay0_sac (raw ep_re) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HalfCheetah | VDPO (Ret_nor) + baselines |  | 1.046±0.215 (3/3) | 0.652±0.179 (3/3) | 0.371±0.039 (3/3) | 0.824±0.139 (3/3) | 0.324±0.064 (3/3) | 0.354±0.047 (3/3) | -246.816±18.707 (3/3) | 7047.470±1413.152 (3/3) |
| Hopper | VDPO (Ret_nor) + baselines |  | 1.227±0.386 (2/3) | 0.902±0.567 (3/3) | 0.557±0.308 (3/3) | 1.220±0.452 (3/3) | 0.427±0.049 (3/3) | 0.152±0.123 (3/3) | 20.440±8.550 (3/3) | 2236.489±807.747 (3/3) |
| Ant | VDPO (Ret_nor) + baselines |  | 1.084±0.558 (3/3) | 0.991±0.143 (3/3) | 0.749±0.162 (3/3) | 1.008±0.157 (3/3) | 0.466±0.285 (3/3) | 0.481±0.008 (3/3) | -43.634±27.458 (3/3) | 4628.016±777.024 (3/3) |
| Walker2d | VDPO (Ret_nor) + baselines |  | 1.050±0.578 (3/3) | 0.385±0.141 (3/3) | 0.289±0.051 (3/3) | 1.300±0.131 (3/3) | 0.178±0.045 (3/3) | 0.085±0.052 (3/3) | 4.955±0.579 (3/3) | 3708.542±498.420 (3/3) |
| Task average | VDPO (Ret_nor) + baselines |  | 1.091±0.469 (11/12) | 0.732±0.393 (12/12) | 0.492±0.251 (12/12) | 1.088±0.318 (12/12) | 0.349±0.187 (12/12) | 0.268±0.173 (12/12) | -66.264±108.257 (12/12) | 4405.129±1982.519 (12/12) |

## Table 2: Raw returns (all columns unnormalized, ep_re at target step)

Same layout as Table 1, but every cell is the **raw** episodic return (`trans_decision ep_re` for VDPO; `FINAL_RESULT ep_re` for baselines).

| Task | Method | Action Delay | 4 | 8 | 16 | 5 | 25 | 50 | random | delay0_sac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HalfCheetah | VDPO + baselines (raw ep_re) |  | 7091.267±488.086 (3/3) | 4290.414±802.811 (3/3) | 2409.574±247.393 (3/3) | 5591.498±438.046 (3/3) | 2044.136±244.098 (3/3) | 2382.151±830.600 (3/3) | -246.816±18.707 (3/3) | 7047.470±1413.152 (3/3) |
| Hopper | VDPO + baselines (raw ep_re) |  | 2229.398±376.762 (2/3) | 1594.763±327.255 (3/3) | 1341.130±858.280 (3/3) | 2406.701±741.487 (3/3) | 991.682±398.439 (3/3) | 268.279±98.967 (3/3) | 20.440±8.550 (3/3) | 2236.489±807.747 (3/3) |
| Ant | VDPO + baselines (raw ep_re) |  | 4601.893±2028.270 (3/3) | 4481.122±388.369 (3/3) | 3337.573±410.403 (3/3) | 4559.456±530.270 (3/3) | 1921.059±1131.637 (3/3) | 2209.644±399.177 (3/3) | -43.634±27.458 (3/3) | 4628.016±777.024 (3/3) |
| Walker2d | VDPO + baselines (raw ep_re) |  | 3608.882±1558.954 (3/3) | 1392.948±448.977 (3/3) | 1086.969±278.422 (3/3) | 4830.323±847.227 (3/3) | 654.062±157.354 (3/3) | 294.047±139.511 (3/3) | 4.955±0.579 (3/3) | 3708.542±498.420 (3/3) |
| Task average | VDPO + baselines (raw ep_re) |  | 4578.629±2203.961 (11/12) | 2939.812±1541.567 (12/12) | 2043.811±1032.074 (12/12) | 4346.995±1353.970 (12/12) | 1402.735±856.334 (12/12) | 1288.530±1112.752 (12/12) | -66.264±108.257 (12/12) | 4405.129±1982.519 (12/12) |

## Regenerate

```bash
python3 extract_ret_nor_combined_md.py --out_md results_ret_nor_v4_1M_delays_combined.md
```

