# Module A2: E2–E4 Replication Overview

> **Status:** Frozen and merged into `master` via PR #2 (`422a5f2`).  
> **Scope:** Paper Section 5 experiments **E2** ($\alpha=0.99$), **E2 Call Spread** ($K_1=100, K_2=101$), **E3** ($\varepsilon=0.01$), **E4-BS**, and **E4-Heston** (Figures 5–11), plus E2/E3 paper-fidelity controls.  
> **Governance:** Scientific scope and acceptance criteria remain governed by [docs/plan.md](../plan.md); repository conventions remain governed by [docs/spec.md](../spec.md).

## 1. Frozen 5-Seed Results Summary

All formal runs use seeds `0–4` (`30,000` training iterations, `batch=1024`, `Adam` cosine `0.005 -> 0.0005`).

| Experiment | Setup | Result (5-Seed Mean ± Std) | Paper Reference | Status |
|---|---|---|---|---|
| **E2** (Fig 6–7) | Heston $d=2$, $\varepsilon=0$, $\alpha=0.50$ vs $0.99$, 1M OOS paths | $p_{0.50}=1.9432 \pm 0.0007$; $p_{0.99}=2.6595 \pm 0.0053$; cross-risk ordering holds 5/5 seeds | $p_{0.50}=1.94$, $p_{0.99}=3.49$ | Qualitative pass; $p_{0.99}$ below `[3.2, 3.8]` |
| **E2-CS** (Fig 8–9) | Call spread ($K_1=100, K_2=101$), $\alpha=0.95$ vs $0.99$, simpler policy, 1M OOS paths | $\alpha=0.99$ lowers $\text{CVaR}_{99}$ by `0.1327 ± 0.0335` in 5/5 seeds; flatter, left-shifted surface | Barrier tightening | Qualitative pass |
| **E3** (Fig 5) | Heston $d=2$, $\varepsilon=0.01$, $\alpha=0.99$, recurrent vs simpler, 1M OOS paths | Recurrent `4.1646 ± 0.0056` vs Simpler `4.2225 ± 0.0040` (recurrent wins 5/5 seeds) | `5.5137` vs `6.7446` | Qualitative pass; narrower gap (`0.0579`) |
| **E2/E3 Fidelity Controls** | Seed-0 arms: `simulator_only`, `paper_input_only`, `simulator_plus_paper_input` | Paper inputs move E3 price level toward paper; no arm recovers the large recurrent–simpler gap | Sensitivity check | Completed |
| **E4-BS** (Fig 10) | BS $d=1$, entropic $\lambda=1$, $\varepsilon \in \{2^{-6}\dots 2^{-10}\}$, 100k OOS paths | Log–log slope `0.7129 ± 0.0017`, $R^2 \approx 0.999$ | `0.67 ± 0.04` | Pass |
| **E4-Heston** (Fig 11) | Heston $d=2$, entropic $\lambda=1$, $\varepsilon \in \{2^{-6}\dots 2^{-10}\}$, 100k OOS paths | Log–log slope `0.6469 ± 0.0173` (median `0.6545`) | `0.71 ± 0.05` | Pass (higher seed sensitivity) |

## 2. Inherited A1 Baseline

Unless a controlled diagnostic documents a deviation (e.g., E2CS uses simpler feed-forward policies on $(S_k, V_k)$ with BatchNorm), A2 inherits the A1 E1 baseline:

- Heston: $S_0 = K = 100$, $V_0 = \theta = 0.04$, $\kappa = 1$, $\sigma = 2$, $\rho = -0.7$, $T = 30/365$, $n = 30$ daily steps
- Instruments: Stock + paper-style variance swap (`out_scale: [1.0, 100.0]`)
- Policy: Semi-recurrent (`hidden=17` for $d=2$, `hidden=16` for $d=1$), `LayerNorm`
- Device: `auto` (records actual `device` and `torch_version` in `metrics.json`)

## 3. Documentation Index

- **Registry & Method:** [experiments-e2-e4.md](experiments-e2-e4.md) · [method_replication-A2.md](method_replication-A2.md)
- **Experiment Reports:** [e2cs_multiseed_summary.md](e2cs_multiseed_summary.md) · [e3_multiseed_summary.md](e3_multiseed_summary.md) · [e4_robustness_analysis.md](e4_robustness_analysis.md)
- **Fidelity & Diagnostics:** [e2e3_fidelity_controls.md](e2e3_fidelity_controls.md) · [e2cs_diagnostics.md](e2cs_diagnostics.md) · [e3_independent_evaluation.md](e3_independent_evaluation.md)
- **Historical Notes:** [plan_A2_next.md](plan_A2_next.md) · [a2_next_phase_implementation.md](a2_next_phase_implementation.md) · [e2_kaggle.md](e2_kaggle.md) · [e4_seed0_analysis.md](e4_seed0_analysis.md)

> **Artifact note:** Historical references to `backups/` and `reports/` in A2 sub-reports refer to the off-repo A2 archive (excluded from `master` per [docs/spec.md](../spec.md) §4 & §7). Canonical figures are committed under `figures/`.

## 4. Reproducing A2 Commands

Run from the repository root (see [scripts/a2/README.md](../../scripts/a2/README.md) for full options):

```bash
# E2 core
python -m src.replication.train --config configs/exp/e2.yaml --seed 0

# E2 Call Spread, E3, E4-BS & E4-Heston
python -m scripts.a2.run_e2cs --config configs/exp/e2cs.yaml --seed 0
python -m scripts.a2.run_e3 --config configs/exp/e3.yaml --seed 0
python -m scripts.a2.run_e4_grid --config configs/exp/e4bs.yaml --seed 0
python -m scripts.a2.run_e4_grid --config configs/exp/e4h.yaml --seed 0

# Batch seed-0 runner & figures
python -m scripts.a2.run_a2_seed0_batch --seed 0 --device auto
python -m scripts.a2.make_figures_a2 --only all --e4-data-root results
```
