# VDPO normalized evaluation returns ($Ret_{nor}$)

## Metric (paper-style normalization)

Define $Ret_{nor} = (Ret_{alg} - Ret_{rand}) / (Ret_{df} - Ret_{rand})$, where:

- $Ret_{alg}$: VDPO return (`trans_decision ep_re`) at the target global step.
- $Ret_{rand}$: random policy baseline return at the same target step.
- $Ret_{df}$: delay-free SAC baseline (`delay0_sac`) at the same target step.

Per-cell statistics are mean ± std over available seeds; `(n/3)` means `n` of `3` seeds contributed after baselines and VDPO logs are both available.

## Data sources

- **VDPO grid logs**: `output/slurm_logs/g1-pickup-grid-search/411547` (array job `411547`).
- **Baselines** (`random`, `delay0_sac`): `output/slurm_logs/vdpo-baselines/411339`.
- **Target global step**: 1,000,000.
- **Environments**: HalfCheetah-v4, Hopper-v4, Ant-v4, Walker2d-v4.
- **Action delays**: [5, 25, 50].
- **Seeds**: [0, 1, 2].

## Incomplete VDPO runs (excluded)

The following array task indices had no `global step 1,000,000, trans_decision ep_re ...` line in their `.out` log and were **skipped** for aggregation:

- task `11` → `Walker2d-v4`, delay `50`, seed `0`
- task `15` → `Hopper-v4`, delay `5`, seed `1`

## Results table

| Task | Method | Action Delay | 5 | 25 | 50 |
| --- | --- | --- | --- | --- | --- |
| HalfCheetah | VDPO (Ret_nor) |  | 1.046±0.215 (3/3) | 0.652±0.179 (3/3) | 0.371±0.039 (3/3) |
| Hopper | VDPO (Ret_nor) |  | 1.227±0.386 (2/3) | 0.902±0.567 (3/3) | 0.557±0.308 (3/3) |
| Ant | VDPO (Ret_nor) |  | 1.084±0.558 (3/3) | 0.991±0.143 (3/3) | 0.749±0.162 (3/3) |
| Walker2d | VDPO (Ret_nor) |  | 1.050±0.578 (3/3) | 0.385±0.141 (3/3) | 0.254±0.017 (2/3) |
| Task average | VDPO (Ret_nor) |  | 1.091±0.469 (11/12) | 0.732±0.393 (12/12) | 0.504±0.258 (11/12) |

## Paper: Table 2 (VDPO only, same layout)

The following values are transcribed from **Table 2** in Wu et al., *Variational Delayed Policy Optimization*, NeurIPS 2024 ([arXiv:2405.14226](https://arxiv.org/abs/2405.14226), PDF Table 2). They report $Ret_{nor}$ with **5 / 25 / 50 constant action delays** and **1M global steps**, same metric definition as in Sec. 4.2.2 of the paper. Only the **VDPO (ours)** column is shown here (other baselines omitted).

| Task | Method | Action Delay | 5 | 25 | 50 |
| --- | --- | --- | --- | --- | --- |
| HalfCheetah | Paper VDPO (Table 2) |  | 1.03±0.08 | 0.70±0.17 | 0.72±0.21 |
| Hopper | Paper VDPO (Table 2) |  | 1.22±0.08 | 0.82±0.40 | 0.22±0.04 |
| Ant | Paper VDPO (Table 2) |  | 1.11±0.04 | 0.56±0.06 | 0.46±0.07 |
| Walker2d | Paper VDPO (Table 2) |  | 1.27±0.04 | 0.27±0.11 | 0.11±0.03 |

## Side-by-side (this run vs paper VDPO)

Numbers are **mean±std** as in the tables above. This run includes `(n/3)` where seeds are partial; the paper table reports std across seeds for their official runs.

| Environment | Delay | This run (job 411547) | Paper Table 2 VDPO |
| --- | ---: | --- | --- |
| HalfCheetah-v4 | 5 | 1.046±0.215 (3/3) | 1.03±0.08 |
| HalfCheetah-v4 | 25 | 0.652±0.179 (3/3) | 0.70±0.17 |
| HalfCheetah-v4 | 50 | 0.371±0.039 (3/3) | 0.72±0.21 |
| Hopper-v4 | 5 | 1.227±0.386 (2/3) | 1.22±0.08 |
| Hopper-v4 | 25 | 0.902±0.567 (3/3) | 0.82±0.40 |
| Hopper-v4 | 50 | 0.557±0.308 (3/3) | 0.22±0.04 |
| Ant-v4 | 5 | 1.084±0.558 (3/3) | 1.11±0.04 |
| Ant-v4 | 25 | 0.991±0.143 (3/3) | 0.56±0.06 |
| Ant-v4 | 50 | 0.749±0.162 (3/3) | 0.46±0.07 |
| Walker2d-v4 | 5 | 1.050±0.578 (3/3) | 1.27±0.04 |
| Walker2d-v4 | 25 | 0.385±0.141 (3/3) | 0.27±0.11 |
| Walker2d-v4 | 50 | 0.254±0.017 (2/3) | 0.11±0.03 |

## Regenerate

```bash
python3 extract_ret_nor_md.py --vdpo_job_id 411547 --baselines_job_id 411339 --target_global_step 1000000 --out_md <path.md> --full_report
```
