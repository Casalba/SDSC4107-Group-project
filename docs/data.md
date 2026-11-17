# B1 Data Documentation: Hong Kong Market Data, Calibration & Scenario Generator

> Deliverable owner: **B1** (`docs/plan.md` §4–§5 & `docs/spec.md` §6 B1 DoD).  
> Target regime: **2010-01-04 to 2024-12-31** (`3,692` official HKEX cash-index trading days; **180** monthly 30-day option windows = `108` train `2010–2018` + `72` OOS test `2019–2024`).  
> Timezone & calendar: Official HKEX trading days (`Asia/Hong_Kong`), no synthetic business-day interpolation.  
> **Academic License**: Raw and processed market series are fetched solely for coursework inside CityU SDSC 4107 and are excluded from Git via `.gitignore` (`data/**`); **do not redistribute raw or processed CSVs**.

---

## 1. `^VHSI` Reconnaissance (Task #1)

**Finding: AVAILABLE (`^HSIL`, 99.97% daily coverage).**

1. **Symbol & Endpoint Discovery**:
   - Direct Yahoo Finance queries for literal `^VHSI` return `HTTP 404 Not Found`, and legacy static Hang Seng Indexes URLs (`hsi.com.hk/static/.../VHSI_d.csv`, `/data/eng/rt/index-series/vhsi/performance.do`) return `HTTP 404 Not Found`.
   - Searching Yahoo Finance's HKG exchange directory locates the official **HSI Volatility Index (`VHSI`)** under symbol **`^HSIL`** (`shortName = "HSI VOLATILITY INDEX"`, currency `HKD`, exchange `HKG`, active since `2003-06-19`).
2. **Crisis Peak Verification**:
   - Observed daily closes of `^HSIL` match historical Hang Seng 30-day ATM implied volatility spikes: `40.27` (`2015-08-25` China crash), `64.80` (`2020-03-16` COVID-19 panic), `49.44` (`2022-03-15`), `43.12` (`2022-10-24`), and `44.50` (`2024-10-07`).
3. **Coverage & Role in `plan.md` §4.4 / §4.7**:
   - Across `2010-01-04..2024-12-31` (`3,692` HKEX trading days), `^HSIL` has `3,691` valid closes (**99.973% coverage**). Only **1** day (`2024-02-19`, `0.027%`) is missing and forward-filled from `2024-02-16` with `is_filled_vhsi = 1`.
   - **B2 Usage**: Main evaluation uses **capital requirement $C^*$** (`C*95`, `C*99`, independent of option premia per `plan.md` §4.7 / §10.2). For the optional premium-adjusted comparison table (`plan.md` §4.4), **B2 should use real `vhsi` (`^HSIL` in annualized decimal IV) directly**, reserving `hsi_vrp{0,3,5}_vol` in `data/interim/hk_vol_forecasts.csv` strictly as a GARCH + VRP sensitivity check.

---

## 2. Raw Data Provenance & Checksums (Task #2)

Fetch date (`asof`): **`2026-11-18`** via reproducible CLI `python scripts/fetch_hk.py --asof 20261118` (window `2009-01-01..2025-02-28` so `2009` provides trailing 365-day dividend yields for `2010-01` and `2025-01..02` provides 30-day terminal settlement prices for the `2024-12` option window). Verified identical SHA-256 digests under `--offline` re-execution (`data/checksums.txt`).

| Series | Source URL / API | Raw File | Rows | Date Span | SHA-256 |
|---|---|---|---|---|---|
| `^HSI` Spot Index | Yahoo Chart v8 (`%5EHSI`) | `data/raw/yahoo/hsi_20261118.csv` | 3,986 | `2009-01-02..2025-02-28` | `8aa798cfaca5ba3819e4c3afcc6c372bdc11fffaad247ea30bdeb70dc2302042` |
| `^HSCE` Spot Index | Yahoo Chart v8 (`%5EHSCE`) | `data/raw/yahoo/hsce_20261118.csv` | 3,986 | `2009-01-02..2025-02-28` | `2e152a61951b26e12e9f06c89e76f81b1aa358033fa9390348b664ea8d3f9a3b` |
| `2800.HK` ETF (TraHK) | Yahoo Chart v8 (`2800.HK`) | `data/raw/yahoo/etf2800_20261118.csv` | 3,986 | `2009-01-02..2025-02-28` | `a1e63bf61c71bd8633dddfdc0f874457163b7b115df2968d121132be6fdfaa92` |
| `2800.HK` Cash Dividends | Yahoo Events (`2013–2024`) + TraHK (`2009–2012`) | `data/raw/yahoo/etf2800_div_20261118.csv` | 32 | `2009-04-29..2024-10-30` | `ee2008f6f2b9f0d330e8295ae44d04720961bf169f2d326ccb217eba8948abad` |
| `^VHSI` (`^HSIL`) | Yahoo Chart v8 (`%5EHSIL`) | `data/raw/yahoo/vhsi_20261118.csv` | 3,986 | `2009-01-02..2025-02-28` | `3396bad8151870caf117c98772038d135fe5fd9b96dcf0996fbba3b97cd97322` |
| HIBOR 1M Fixing | HKMA Open API (`segment=hibor.fixing`) | `data/raw/hkma/hibor1m_20261118.csv` | 7,473 | `1996-07-01..2026-10-30` | `d40078e9f3430840bd9259fce29870d70a7efc90532d150d6c982a6bfb5f2dff` |
| Processed Daily Table | `src.hk.data.build_processed_datasets` | `data/processed/hk_daily.csv` | 3,692 | `2010-01-04..2024-12-31` | `0dff67b65ea8b392f20196f10969b7e1f0ddc5859929a9d0d4ff79213f2ba917` |
| Extended Audit Table | `src.hk.data.build_processed_datasets` | `data/interim/hk_daily_full_with_audit.csv` | 3,980 | `2009-01-02..2025-02-28` | `2999f7bfd103cc80bf9ed5ec0036a5c1fecbd572b6b95fe70ad2af3edf702bf3` |

---

## 3. Alignment, Dividend Adjustment & Missing-Value Audit (Task #3)

- **Schema of `data/processed/hk_daily.csv`** (exact 7 columns, zero `NaN`/`Inf`):
  - `date` (`YYYY-MM-DD`, Asia/Hong_Kong HKEX cash-index trading days)
  - `hsi` (`^HSI` closing index points)
  - `hsce` (`^HSCE` closing index points)
  - `etf2800` (`2800.HK` backward total-return adjusted price in HKD, reinvesting all 32 semi-annual cash dividends via $A_t = A_{t+1}(1 - D_{t+1}/C_t)$ across ex-dividend dates; avoids artificial price drops at ex-dates when computing ETF hedge P&L)
  - `div_yield` (annualized decimal trailing dividend yield $q_t = \sum_{s \in (t-365\text{d}, t]} D_s / C_t^{\text{raw}}$; mean `3.17%`, range `0.63%–7.90%`)
  - `hibor1m` (annualized decimal HKMA 1-month HIBOR fixing `ir_1m / 100`; mean `1.10%`, range `0.06%–5.66%`)
  - `vhsi` (annualized decimal `^HSIL` close `/ 100`; mean `21.28%`, range `11.36%–64.80%`)
- **HKEX Calendar & Gap Handling**:
  - Official cash-index calendar (`^HSI` non-null dates) automatically excludes Saturdays (`2011-12-24`, `2011-12-31`), Hong Kong public holidays, and full-day HKEX **Typhoon Signal No. 8 / 10 closures** (`2013-08-14`, `2016-08-02`, `2016-10-21`, `2017-08-23`).
  - Forward-fill audit flags over `2010–2024`: `is_filled_hsi = 0`, `is_filled_hsce = 0`, `is_filled_etf2800 = 0`, `is_filled_hibor1m = 0`, `is_filled_vhsi = 1` (`2024-02-19`, `0.027%`). `is_halted_2800 = 1` flags `21` days where Yahoo reports `volume = 0` on `2800.HK` despite valid closes.
- **Pre-2013 `2800.HK` Dividend Supplement**: Yahoo's dividend feed begins on `2013-04-29`; we supplement the 8 official Hang Seng Investment Management / TraHK distributions (`2009-04-29: 0.16`, `2009-10-29: 0.38`, `2010-04-29: 0.12`, `2010-10-29: 0.46`, `2011-04-28: 0.13`, `2011-10-31: 0.50`, `2012-04-27: 0.14`, `2012-10-31: 0.52` HKD/unit).
- **Synthetic Futures Construction (`spec` §4.3)**: Since continuous historical HSI futures contracts are unavailable publicly (documented limitation), `src.hk.data.synthetic_futures_path` builds $F_k = S_k \exp((r_k - q_k)\tau_k)$ with $\tau_k = \frac{30}{365}(1 - k/n)$, and strikes are rounded to the nearest `100` index points (`round_hsi_strike`).

---

## 4. HK Trading Cost & Contract Specification (`plan.md` §4.5 v1.4)

Mapped strictly under paper Eq. (5.7) (`c_k = eps * |delta_k - delta_{k-1}| * S_k`) on the underlying leg (`spec.md` §5.1 / `configs/hk/*.yaml`):

| Config File | Instrument | One-Way `eps` | Round-Trip Cost | Fee Structure & Exchange Frictions |
|---|---|---|---|---|
| `configs/hk/fut1bp.yaml` | Synthetic HSI Futures | `0.00005` (`0.5 bp`) | **`~1 bp`** | Commission ~HK\$8/contract/side (~0.4 bp on HK\$1M notional) + 0.25–0.5 tick spread (~0.5 bp/tick at HK\$50/pt); 0% stamp duty |
| `configs/hk/fut10bp.yaml` | Stress Futures Tier | `0.00050` (`5.0 bp`) | **`10 bp`** | Mid-friction sensitivity tier bridging futures and ETF slippage |
| `configs/hk/etf30bp.yaml` | `2800.HK` Tracker Fund | `0.00150` (`15.0 bp`) | **`~30 bp` (`25–40 bp`)** | Commission 6–10 bp + HKEX/SFC/AFRC levies ~1.7 bp + 0.5–1 tick bid–ask spread (~20 bp); **0% stamp duty** (HK ETFs exempt since `2015-02-13`, IRD FAQ); 500-share board lot (`~90–100` lots per HK\$1M call) |

---

## 5. Stylized Facts, Jump Tests & Crisis Regimes (Task #4)

> **One-sentence health-check conclusions**:
> - *Stylized facts*: Both `^HSI` and `^HSCE` exhibit strong volatility clustering ($p < 10^{-149}$) and pronounced fat tails (excess kurtosis `3.55` and `3.35`, Jarque–Bera $p = 0$).
> - *Jump tests*: BNS bipower variation attributes `8.85%` (`^HSI`) and `6.58%` (`^HSCE`) of quadratic variation to discontinuous jumps, with extreme single-day tail moves exceeding `3.5σ` in `0.72%–0.74%` of days.
> - *Regimes*: Realized volatility surges during `2015H2`, `2020H1`, and `2022` (`24%–38%` vs. `19%–23%` non-crisis), driving `2019–2024` OOS volatility (`23.81%`) well above `2010–2018` (`18.31%`).

### 5.1 Distributional & Dependence Statistics (`2010-01-04..2024-12-31`, $N = 3,691$ daily log-returns)

| Index | Sample | Ann. Mean | Ann. Vol | Min / Max Day | Skewness | Ex. Kurtosis | JB Stat ($p$) | LB $Q(10)$ on $r_t$ ($p$) | LB $Q(10)$ on $r_t^2$ ($p$) |
|---|---|---|---|---|---|---|---|---|---|
| `^HSI` | Full (`2010–2024`) | `-0.58%` | `20.68%` | `-9.88% / +8.69%` | `-0.074` | `3.55` | `1,932.9` (`0.00`) | `10.87` (`0.368`) | `729.9` (`2.3e-150`) |
| `^HSI` | Train (`2010–2018`) | `+1.92%` | `18.31%` | `-6.02% / +5.52%` | `-0.318` | `2.33` | `533.7` (`1.3e-116`) | `10.07` (`0.435`) | `371.6` (`1.0e-73`) |
| `^HSCE` | Full (`2010–2024`) | `-3.82%` | `24.59%` | `-10.72% / +11.78%` | `+0.096` | `3.35` | `1,724.5` (`0.00`) | `14.76` (`0.141`) | `763.3` (`1.6e-157`) |
| `^HSCE` | Train (`2010–2018`) | `-2.62%` | `23.21%` | `-6.52% / +7.82%` | `-0.063` | `1.96` | `352.1` (`3.5e-77`) | `11.68` (`0.307`) | `496.3` (`2.7e-100`) |

### 5.2 Non-Parametric Jump Tests (Lee–Mykland 2008 & Barndorff-Nielsen–Shephard)

| Index | Realized Vol | Bipower Vol | BNS Jump Share | LM Gumbel $\alpha=1\%$ ($L > 6.00$) | Pointwise ($L > 3.5$) | Largest Jump Events (`date`: ret, $L$) |
|---|---|---|---|---|---|---|
| `^HSI` | `20.68%` | `19.74%` | `8.85%` | `1` (`0.028%`) | `26` (`0.716%`) | `2018-02-06` (`-5.25%`, $L=6.75$), `2024-10-08` (`-9.88%`, $5.88$), `2022-03-16` (`+8.69%`, $5.35$) |
| `^HSCE` | `24.58%` | `23.75%` | `6.58%` | `0` (`0.000%`) | `27` (`0.744%`) | `2022-03-16` (`+11.78%`, $L=5.97$), `2024-10-08` (`-10.72%`, $5.90$), `2018-02-06` (`-6.06%`, $5.44$) |

### 5.3 Stress Regimes vs. Train/Test Split

| Regime Window | Trading Days | `^HSI` Ann. Ret / Vol | `^HSI` MaxDD (Worst Day) | `^HSCE` Ann. Ret / Vol | `^HSCE` MaxDD (Worst Day) |
|---|---|---|---|---|---|
| `2010–2018` (Train) | 2,215 | `+1.92% / 18.31%` | `-35.59% (-6.02%)` | `-2.62% / 23.21%` | `-49.29% (-6.52%)` |
| `2019–2024` (OOS Test) | 1,476 | `-4.33% / 23.81%` | `-52.75% (-9.88%)` | `-5.61% / 26.53%` | `-59.61% (-10.72%)` |
| `2015H2` (China Deval Crash) | 126 | `-36.10% / 24.16%` | `-21.79% (-6.02%)` | `-59.08% / 28.24%` | `-28.80% (-6.28%)` |
| `2020H1` (COVID-19 Shock) | 122 | `-29.59% / 28.18%` | `-25.33% (-5.72%)` | `-27.87% / 28.70%` | `-25.05% (-4.62%)` |
| `2022` (Rate Hike & Tech Drawdown) | 246 | `-17.20% / 32.53%` | `-41.17% (-6.57%)` | `-21.07% / 37.76%` | `-43.82% (-7.58%)` |
| Non-Crisis Other Days | 3,197 | `+3.21% / 18.96%` | `-54.87% (-9.88%)` | `+0.61% / 22.93%` | `-66.21% (-10.72%)` |

Figures generated by `python -m scripts.make_hk_b1_figures`:
- `figures/fig_hk_stylized_facts.png` (standardised tail density vs. $\mathcal{N}(0,1)$ & ACF volatility clustering)
- `figures/fig_hk_garch_vol.png` (2010–2024 walk-forward GARCH+Jump conditional volatility vs. 21-day realized vol and `^VHSI`)

---

## 6. Strict Walk-Forward GARCH(1,1) + Jump Calibration (Task #5)

Estimated via multi-start MLE **exclusively on `2010-01-04..2018-12-31` ($N = 2,215$ daily pct returns $y_t = 100 \ln(S_t/S_{t-1})$)**; standard errors are Bollerslev–Wooldridge QMLE sandwich estimates mapped via Delta Method. Outputs are written to `results/calib/hsi_garch.json`, `results/calib/hsce_garch.json`, and `data/interim/hk_vol_forecasts.csv`.

| Index | Model | $\mu$ (SE) | $\omega$ (SE) | $\alpha$ (SE) | $\beta$ (SE) | $\lambda$ (SE) | $\mu_J$ (SE) | $\sigma_J$ (SE) | LogLik (AIC) | LRT $p$ vs Plain | LB $\hat{z}$ $p_{10}$ / $\hat{z}^2$ $p_{10}$ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `^HSI` | GARCH(1,1) | `0.0357` (`0.0218`) | `0.0203` (`0.0089`) | `0.0494` (`0.0108`) | `0.9349` (`0.0148`) | — | — | — | `-3323.30` (`6654.6`) | — | `0.421 / 0.454` |
| `^HSI` | **GARCH+Jump** | `0.0274` (`0.0215`) | `8.3e-7` (`1e-6`) | `0.0405` (`0.0069`) | `0.9398` (`0.0108`) | `0.381` (`0.135`) | `-0.301` (`0.114`) | `0.980` (`0.115`) | **`-3282.64` (`6579.3`)** | **`1.59e-17`** | **`0.435 / 0.202`** |
| `^HSCE` | GARCH(1,1) | `0.0170` (`0.0279`) | `0.0302` (`0.0136`) | `0.0500` (`0.0102`) | `0.9352` (`0.0144`) | — | — | — | `-3847.34` (`7702.7`) | — | `0.189 / 0.239` |
| `^HSCE` | **GARCH+Jump** | `0.0150` (`0.0277`) | `8.3e-7` (`1e-6`) | `0.0439` (`0.0087`) | `0.9399` (`0.0126`) | `0.424` (`0.231`) | `-0.090` (`0.103`) | `1.139` (`0.222`) | **`-3829.46` (`7672.9`)** | **`8.47e-08`** | **`0.199 / 0.137`** |

*Note*: Likelihood-ratio tests overwhelmingly reject Gaussian GARCH(1,1) in favor of Poisson-Gaussian GARCH(1,1)+Jump ($p = 1.6\times 10^{-17}$ for `^HSI`, $p = 8.5\times 10^{-8}$ for `^HSCE`). Under constant Poisson intensity $\lambda$, long-run baseline variance is sustained by the constant jump variance floor $\lambda(\mu_J^2 + \sigma_J^2)$, driving diffusive intercept $\omega$ toward the non-negative boundary (`~1e-6`) with significant negative jump asymmetry on `^HSI` ($\mu_J = -0.301\%$, $t = -2.63$). Standardized residuals and squared standardized residuals show **no remaining autocorrelation** (`LB p10 >= 0.137`, `p20 >= 0.095`).

---

## 7. Scenario Generator & B1 -> B2 Integration Interface (Tasks #6 & #7)

- **Scenario Generator (`src/hk/synth.py`)**:
  - `HKMarket` + `HKWorld` mirror A1's `BSWorld` / `HestonWorld` (`simulate`, `features`, `prices`, `payoff`, `cost`, `terminal_loss`) in pure `float64` PyTorch tensors seeded via `seed_everything()`.
  - `mode="garch"` (M2 primary calibrated GARCH(1,1)+Jump with 250-step stationary burn-in), `mode="garch_plain"`, and `mode="bootstrap"` (circular 5-day block bootstrap on `2010–2018` training returns).
  - Unit tests (`tests/test_hk_synth.py`) confirm simulated daily mean, variance (`< 5%` relative error vs. analytical stationary variance at $60{,}000$ paths), and exact step-by-step compatibility with A1's frozen `src.policy.heston_d1.load()`.
- **B2 Usage Snippet**:
  ```python
  from src.hk.data import extract_monthly_windows, load_hk_daily, window_features
  from src.hk.synth import HKMarket, HKWorld
  from src.policy.heston_d1 import load as load_heston_d1

  windows = extract_monthly_windows(index_col="hsi")  # 180 windows: 108 train, 72 test
  world = HKWorld(HKMarket(index="hsi", mode="garch", eps=0.0015), device="cpu")
  paths = world.simulate(n_paths=1024)
  feats = world.features(paths)  # shape (1024, 30, 2) == [log(S_k/S_0), t_k/T]
  policy = load_heston_d1(device="cpu")
  loss = world.terminal_loss(policy.forward_path(feats), paths)
  ```
- **Reproducibility Commands**:
  ```powershell
  .\.venv\Scripts\python.exe scripts/fetch_hk.py --asof 20261118 --offline
  .\.venv\Scripts\python.exe -m src.hk.calibrate
  .\.venv\Scripts\python.exe -m scripts.make_hk_b1_figures
  .\.venv\Scripts\python.exe scripts/ci.py
  ```
