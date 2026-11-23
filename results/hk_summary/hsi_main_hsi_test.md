# HK Cross-Validation Results Summary (HSI | test)

- **Index**: `hsi` | **Split**: `test` (72 monthly windows)

## 1. Main Table: Capital Requirement C* & Trading Metrics

| Cost Leg | Strategy | C*95 | C*99 | Mean Loss | Turnover | Cost / Notional |
|---|---|---:|---:|---:|---:|---:|
| fut1bp | Bench (Delta) | 5.9733 | 7.1418 | 2.9316 | 2.1644 | 0.00011 |
| fut1bp | M1 (Zero-shot) | 7.3162 | 8.0396 | 2.8336 | 2.8030 | 0.00014 |
| fut1bp | M2 (Re-fit, α=0.5) | 4.9857 ± 0.2413 | 5.8050 ± 0.2428 | 2.7265 ± 0.0371 | 2.2372 | 0.00011 |
| fut1bp | M2 (Re-fit, α=0.99) | 5.8265 ± 0.0464 | 7.3090 ± 0.1615 | 2.9740 ± 0.0167 | 1.9628 | 0.00010 |
| fut10bp | Bench (Delta) | 6.0650 | 7.2877 | 3.0288 | 2.1644 | 0.00108 |
| fut10bp | M1 (Zero-shot) | 7.4979 | 8.1394 | 2.9618 | 2.8030 | 0.00143 |
| etf30bp | Bench (Delta) | 5.8447 | 7.2811 | 3.0339 | 2.1741 | 0.00326 |
| etf30bp | M1 (Zero-shot) | 7.4852 | 8.0988 | 3.0530 | 2.7990 | 0.00428 |
| etf30bp | M2 (Re-fit, α=0.5) | 6.6796 ± 1.0956 | 8.4265 ± 1.9208 | 2.7753 ± 0.0615 | 1.4383 | 0.00216 |
| etf30bp | M2 (Re-fit, α=0.99) | 5.9726 ± 0.0417 | 7.7491 ± 0.1177 | 3.0068 ± 0.0132 | 1.7416 | 0.00261 |

## 2. Hypothesis Tests (P1–P4)

- **P1_futures_1bp_parity**: `{"point": -0.5750834329792056, "ci_low": -1.1234800340538031, "ci_high": 0.46431840211288705, "ci_excludes_zero": false, "p1_within_5pct": false, "p2_strictly_better": false}`
- **P2_etf_30bp_advantage**: `{"point": 1.8366808214027053, "ci_low": 0.4653877404777395, "ci_high": 4.26082956206497, "ci_excludes_zero": true, "p1_within_5pct": false, "p2_strictly_better": false}`
- **P3_risk_aversion_crossover_etf**: `{"c95_a99_minus_a50": -0.7070139828477036, "c99_a99_minus_a50": -0.6774236291718729}`
- **P4_refit_vs_zeroshot_etf**: `{"point": 0.19618549769029414, "ci_low": -1.365383219372835, "ci_high": 1.869674370051551, "ci_excludes_zero": false, "p1_within_5pct": true, "p2_strictly_better": false}`
