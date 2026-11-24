# SDSC4107 · Deep Hedging Replication & Hong Kong Cross-Validation

[![CI](https://github.com/Casalba/SDSC4107-Group-project/actions/workflows/ci.yml/badge.svg)](https://github.com/Casalba/SDSC4107-Group-project/actions/workflows/ci.yml)

Empirical replication of Buehler, Gonon, Teichmann, and Wood (2019), *Deep Hedging* (Section 5: Experiments E1–E4), extended with an out-of-sample empirical cross-validation on the Hong Kong equity derivatives market (`^HSI`, `^HSCE`, `2800.HK`, `^VHSI`, 2010–2024) under realistic execution frictions.

> **Status:** All four modules (**A1, A2, B1, B2**) are complete and merged into `master`.

---

## Project Overview

This repository answers two linked questions:

1. **Paper Replication (Part A — E1–E4):** Can semi-recurrent neural hedging policies reproduce the quantitative and qualitative findings of *Deep Hedging* Section 5 under Heston ($d=2$, stock + variance swap) and Black–Scholes ($d=1$) worlds—recovering model delta at zero cost (E1), exhibiting risk-aversion trade-offs (E2), demonstrating architecture advantages under transaction costs (E3), and obeying the $O(\varepsilon^{2/3})$ asymptotic pricing law (E4)?
2. **Hong Kong Empirical Cross-Validation (Part B — B1 & B2):** Under realistic Hong Kong frictions—synthetic HSI futures (~1 bp round-trip) and Tracker Fund `2800.HK` ETF (~30 bp round-trip + 500-share board lots, stamp-duty exempt)—does Deep Hedging reduce the zero-initial-capital tail requirement $C^*_\alpha = \text{CVaR}_\alpha(L_T)$ relative to an analytical `^VHSI`-driven Black–Scholes Delta benchmark, and does local `GARCH(1,1)+Jump` re-fitting outperform zero-shot transfer from Heston?

### Module Ownership & Reports

| Module | Scope | Primary Report / Docs | Status |
|---|---|---|---|
| **A1** | Repo skeleton, `pfhedge` adapters, B0 benchmark checks, E1 (5 seeds, Fig 2–4), frozen `d=1` transfer policy | [docs/report_a1.md](docs/report_a1.md), [docs/method_replication.md](docs/method_replication.md), [docs/b0.md](docs/b0.md) | Merged |
| **A2** | E2 ($\alpha=0.99$), E2 Call Spread, E3 ($\varepsilon=0.01$), E4-BS & E4-Heston (5 seeds, Fig 5–11), fidelity controls | [README_A2.md](README_A2.md), [docs/A2/experiments-e2-e4.md](docs/A2/experiments-e2-e4.md), [docs/A2/method_replication-A2.md](docs/A2/method_replication-A2.md) | Merged |
| **B1** | HK data pipeline (`^HSI`, `^HSCE`, `2800.HK`, dividends, 1M HIBOR, `^VHSI`), stylized facts, `GARCH(1,1)+Jump` calibration, `HKWorld` simulator | [docs/data.md](docs/data.md) | Merged |
| **B2** | Monthly rolling HK backtester, 500-share board-lot rounding, circular/paired block bootstrap, M1/M2 transfer & crisis/VRP/HSCE analyses | [docs/hk_empirical_cross_validation_report_b2.md](docs/hk_empirical_cross_validation_report_b2.md) | Merged |

---

## Key Results at a Glance

### Part A: Paper Replication (5 Seeds, 0–4)

| Exp | Setup | Our Result (5-Seed Mean ± Std) | Paper Target | Outcome |
|---|---|---|---|---|
| **B0** | Analytical Heston / VarSwap / Mid-point vs. 100k MC | All 3 checks `< 0.5%` | `< 1.5% / 1% / 1%` | Pass |
| **E1** | Heston $d=2$, $\varepsilon=0$, $\alpha=0.50$ (Fig 2–4) | Price `1.9431 ± 0.0007`; mean loss `1.6913` vs $q=1.6918$; $\delta_1$ RMSE `0.094–0.113` | Price `1.94`, $q=1.69$ | Pass |
| **E1-d1** | Heston $d=1$, $\varepsilon=0$, $\alpha=0.50$ (A1→B2 handoff) | Price `2.5338 ± 0.007` (10% below analytical $\delta_1$ hedge `2.8383`) | Handoff artifact | Pass |
| **E2** | Heston $d=2$, $\alpha=0.50$ vs $0.99$, 1M OOS paths (Fig 6–7) | $p_{0.50}=1.9432 \pm 0.0007$, $p_{0.99}=2.6595 \pm 0.0053$; cross-risk ordering holds 5/5 seeds | $p_{0.99} \in [3.2, 3.8]$ | Qualitative pass; $p_{0.99}$ below band |
| **E2-CS** | Call spread ($K_1=100, K_2=101$), $\alpha=0.95$ vs $0.99$ (Fig 8–9) | $\alpha=0.99$ lowers $\text{CVaR}_{99}$ by `0.1327 ± 0.0335` in 5/5 seeds | Barrier tightening | Qualitative pass |
| **E3** | Heston $d=2$, $\varepsilon=0.01$, $\alpha=0.99$ (Fig 5) | Recurrent `4.1646 ± 0.0056` vs Simpler `4.2225 ± 0.0040` (recurrent wins 5/5 seeds) | `5.51` vs `6.74` | Qualitative pass; narrower gap |
| **E4-BS** | BS $d=1$, entropic $\lambda=1$, $\varepsilon \in \{2^{-6}\dots 2^{-10}\}$ (Fig 10) | Log–log slope `0.7129 ± 0.0017`, $R^2 \approx 0.999$ | `0.67 ± 0.04` | Strong power-law scaling |
| **E4-H** | Heston $d=2$, entropic $\lambda=1$, $\varepsilon \in \{2^{-6}\dots 2^{-10}\}$ (Fig 11) | Log–log slope `0.6469 ± 0.0173` (median `0.6545`) | `0.71 ± 0.05` | Approximate $2/3$ scaling |

### Part B: Hong Kong Empirical Cross-Validation (`^HSI`, 2010–2024)

Calibrated strictly on 2010–2018 (`GARCH(1,1)+Jump`, LRT $p = 1.6 \times 10^{-17}$, `^VHSI` 99.97% covered) and evaluated across **180 full-sample monthly windows** (2010–2024) and **72 out-of-sample test windows** (2019–2024):

| Cost Leg | Strategy | Full Sample $C^*_{95}$ | Full Sample $C^*_{99}$ | OOS (2019–24) $C^*_{95}$ | Turnover |
|---|---|---:|---:|---:|---:|
| `fut1bp` (1 bp RT) | Bench (`^VHSI` Delta) | 5.1215 | 6.4768 | 5.9733 | 2.1423 |
| `fut1bp` (1 bp RT) | M1 (Zero-Shot Heston) | 6.7222 | 7.7330 | 7.3162 | 2.7685 |
| `fut1bp` (1 bp RT) | **M2 (Re-fit, $\alpha=0.50$)** | **4.6550 ± 0.1738** | **5.5378 ± 0.3904** | **4.9857 ± 0.2413** | 2.1367 |
| `fut1bp` (1 bp RT) | M2 (Re-fit, $\alpha=0.99$) | 5.1240 ± 0.0489 | 6.6597 ± 0.1076 | 5.8265 ± 0.0464 | **1.8578** |
| `etf30bp` (30 bp + lots) | **Bench (`^VHSI` Delta)** | **5.2562** | **6.7040** | **5.8447** | 2.1487 |
| `etf30bp` (30 bp + lots) | M1 (Zero-Shot Heston) | 7.0889 | 7.8258 | 7.4852 | 2.7643 |
| `etf30bp` (30 bp + lots) | M2 (Re-fit, $\alpha=0.50$) | 5.7349 ± 0.6984 | 7.7727 ± 1.2383 | 6.6796 ± 1.0956 | **1.4076** |
| `etf30bp` (30 bp + lots) | M2 (Re-fit, $\alpha=0.99$) | 5.4231 ± 0.0436 | 6.8969 ± 0.0893 | 5.9726 ± 0.0417 | 1.6491 |

- **P1 (Low-cost parity):** Confirmed on `fut1bp`—`M2 (α=0.50)` matches or improves upon `Bench` (`C*95` 4.66 vs. 5.12; OOS $\Delta C^*_{99}$ 95% CI strictly below zero).
- **P2 (30 bp ETF trade-off):** `M2 (α=0.50)` cuts turnover by **34.5%** and achieves the lowest mean loss across all VRP tiers, while `M2 (α=0.99)` restores tail parity with `Bench` without requiring forward-looking `^VHSI` inputs.
- **P3 (Risk aversion under friction):** Raising $\alpha$ from `0.50` to `0.99` on `etf30bp` lowers both $C^*_{95}$ (`-0.31`) and $C^*_{99}$ (`-0.88`) and cuts cross-seed std by **93.8%–96.2%**.
- **P4 (Local re-fitting vs. zero-shot):** Jump-free Heston `M1` degrades severely on HK data; `M2 (α=0.99)` lowers $C^*_{95}$ vs. `M1` by `-1.68` on `etf30bp` (95% CI `[-2.50, -0.85]`).
- **Board-lot rounding:** 500-share discretization on `2800.HK` induces negligible tracking error (`0.017–0.021` mean abs loss diff per $S_0 = 100$ notional).

---

## Repository Structure

```text
project/
├─ README.md                    # Project overview (this file)
├─ README_A2.md                 # Detailed A2 progress & command index
├─ pyproject.toml               # pytest / ruff configuration (line-length = 100)
├─ requirements.lock.txt        # Pinned dependency versions
├─ configs/
│   ├─ exp/                     # debug, e1, e1d1, e2, e2cs, e3, e4bs, e4h
│   ├─ fidelity/e2e3/           # Controlled A2 fidelity-arm configs
│   └─ hk/                      # fut1bp.yaml, fut10bp.yaml, etf30bp.yaml
├─ src/
│   ├─ common/                  # config, seed, logging, metrics (CVaR/C*/cost), types
│   ├─ policy/                  # Frozen Policy protocol + weights/heston_d1.pt (A1→B2)
│   ├─ replication/             # env_heston, env_bs, net_recurrent, losses, pricing, benchmark, e1–e4
│   └─ hk/                      # data, calibrate, synth (HKWorld), backtest, stats, train_hk, run_experiments
├─ scripts/
│   ├─ ci.py                    # Local CI mirror (ruff check + ruff format + pytest)
│   ├─ check_b0.py              # B0 three-way analytical vs. MC verification
│   ├─ make_figures.py          # A1 Figures 2–4
│   ├─ export_policy_d1.py      # Export frozen d=1 transfer policy for B2
│   ├─ fetch_hk.py              # B1 HK market data fetcher
│   ├─ make_hk_b1_figures.py    # B1 stylized-facts & GARCH volatility figures
│   ├─ make_hk_b2_figures.py    # B2 capital, turnover frontier, & crisis-regime figures
│   └─ a2/                      # A2 experiment runners, evaluators, diagnostics, & plotters
├─ figures/                     # Publication-ready 300 dpi figures (Fig 2–11 + HK B1/B2)
├─ results/hk_summary/          # Compact HK B2 JSON/Markdown summary tables
├─ docs/                        # plan.md, spec.md, experiments.md, module reports, paper/
└─ tests/                       # Unit & integration tests (A1, A2, B1, B2)
```

> Raw market data (`data/`) and large training outputs/checkpoints (`results/`, `*.npz`, `*.pt`) are gitignored per [docs/spec.md](docs/spec.md) §4 & §7, with explicit exceptions for `src/policy/weights/heston_d1.pt` and `results/hk_summary/`.

---

## Quick Start

### 1. Environment Setup

Python **3.10+** (tested on Python 3.12). Note that `pfhedge==0.23.0` requires `numpy<2.0` (`numpy==1.26.4`, `scipy==1.13.1`).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate  |  Linux/macOS: source .venv/bin/activate
pip install -r requirements.lock.txt
```

All training and evaluation scripts use `device: auto` (CUDA if available, otherwise CPU) and record `device` and `torch_version` in every `metrics.json`.

### 2. Verify Installation & Run CI Locally

```bash
# Run full lint + format + test suite (mirrors GitHub Actions)
python scripts/ci.py

# Or run directly:
ruff check .
ruff format --check .
pytest -q

# Fast end-to-end smoke run
python -m src.replication.train --config configs/exp/debug.yaml
```

---

## Reproducing Experiments

Run all commands from the repository root:

### Part A: Paper Replication

```bash
# A1: B0 benchmark checks, E1 (seeds 0–4), d=1 handoff policy, Figures 2–4
python -m scripts.check_b0
python -m src.replication.train --config configs/exp/e1.yaml --seed 0
python -m scripts.export_policy_d1
python -m scripts.make_figures

# A2: E2, E2 Call Spread, E3, E4-BS & E4-Heston (see scripts/a2/README.md for full options)
python -m scripts.a2.run_a2_seed0_batch
python -m scripts.a2.run_e2cs
python -m scripts.a2.run_e3
python -m scripts.a2.run_e4_grid
python -m scripts.a2.make_figures_a2
```

### Part B: Hong Kong Empirical Cross-Validation

```bash
# B1: Fetch HK data, calibrate GARCH(1,1)+Jump (2010–2018), generate B1 figures
python -m scripts.fetch_hk
python -m src.hk.calibrate
python -m scripts.make_hk_b1_figures

# B2: Train M2 local re-fit policies (seeds 0–4), run HSI/HSCE backtests, generate B2 figures
python -m src.hk.train_hk --config configs/hk/fut1bp.yaml --alpha 0.50
python -m src.hk.train_hk --config configs/hk/etf30bp.yaml --alpha 0.99
python -m src.hk.run_experiments --index hsi --split all --tag hsi_main
python -m src.hk.run_experiments --index hsi --split test --tag hsi_main
python -m src.hk.run_experiments --index hsce --split all --tag hsce_robustness
python -m scripts.make_hk_b2_figures
```

---

## Documentation Index

- **Governance & Standards:** [docs/plan.md](docs/plan.md) (scientific scope & acceptance criteria) · [docs/spec.md](docs/spec.md) (interfaces, naming, DoD) · [docs/experiments.md](docs/experiments.md) (experiment registry)
- **Part A (Replication):** [docs/report_a1.md](docs/report_a1.md) · [docs/method_replication.md](docs/method_replication.md) · [docs/e1_diagnosis.md](docs/e1_diagnosis.md) · [README_A2.md](README_A2.md) · [docs/A2/experiments-e2-e4.md](docs/A2/experiments-e2-e4.md)
- **Part B (Hong Kong):** [docs/data.md](docs/data.md) · [docs/hk_empirical_cross_validation_report_b2.md](docs/hk_empirical_cross_validation_report_b2.md)
