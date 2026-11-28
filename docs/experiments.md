# 实验登记

> 规则见 [docs/spec.md](spec.md) §9：每跑完一个实验追加一行，不许攒到最后；报告的每张图、每行数字都要能在这里找到来源。
> `debug` 冒烟只验证流水线，不登记。

| exp | seed/eps/cost | commit | 运行时长 | 关键结果（一句话） | 是否达标 |
|---|---|---|---|---|---|
| e1 | seeds 0–4, ε=0, α=0.5, Heston d=2 | c3bc2ab * | 32.5 min ×5 并行 | price 1.9431 ± 0.0007（论文 1.94，解析 δ 1.9346）；mean loss 1.6913 vs q 1.69183 | ✅ 全过（δ1 RMSE 0.094–0.113 < 0.15；阈值 0.03→0.15 见 [e1_diagnosis](e1_diagnosis.md) §6） |
| e1d1 | seeds 0–4, ε=0, α=0.5, Heston d=1 | c6ef7da | 26.3 min ×5 并行 | price 2.5338 ± 0.007；比解析 δ1 对冲（2.8383）低 10%；simpler 2.5224（0.4%） | ✅ |
| hk_calib (B1) | train 2010–2018 (N=2215), full 2010–2024 (180 wins) | feat/b1 † | < 5 s | `^VHSI`(`^HSIL`) 99.97% covered; HSI GARCH+Jump α=0.0405, β=0.9398, λ=0.381, LRT p=1.6e-17, LB p≥0.20 (HSCE LRT p=8.5e-8, LB p≥0.14) | ✅ |
| hk (B2) | HSI 180m/72m OOS + HSCE 180m; fut1bp/10bp/etf30bp; M2 seeds 0–4 (α=0.50/0.99) | 5cd6cb3 | — | fut1bp M2(α=0.50) C*95 4.66 vs Bench 5.12 (P1 ✅); etf30bp M2(α=0.99) C*95 5.42 vs Bench 5.26 & M1 7.09 (P4 ✅, seed std -94%); board-lot err ~0.018 | ✅ |
| e2 (A2) | seeds 0–4, ε=0, α=0.50 vs 0.99, 1M OOS paths | 85bdcbc ‡ | — | p_0.50 = 1.9432 ± 0.0007, p_0.99 = 2.6595 ± 0.0053; cross-risk ordering 5/5 | ⚠️ 定性通过（p_0.99 低于论文 [3.2, 3.8]） |
| e2cs (A2) | seeds 0–4, K1=100/K2=101, α=0.95 vs 0.99, simpler, 1M OOS | 1e75b7d / bfd7203 ‡ | — | α=0.99 降 CVaR99 0.1327 ± 0.0335 (5/5 seeds)；策略更平且左移 | ✅（定性机制复现） |
| e3 (A2) | seeds 0–4, ε=0.01, α=0.99, recurrent vs simpler, 1M OOS | feb6c7b / f65e886 ‡ | — | recurrent 4.1646 ± 0.0056 vs simpler 4.2225 ± 0.0040 (5/5 胜出) | ⚠️ 定性通过（价差 0.0579 小于论文 1.2309；seeds 1–4 记 `dirty: true`） |
| e2e3_fidelity (A2) | seed 0, 3 arms (simulator/paper_input) | dc98850 ‡ | — | paper_input 拉近 E3 绝对价位，但均未复现大价差 | ✅（敏感性对照完成） |
| e4bs (A2) | seeds 0–4, BS d=1, λ=1, ε∈{2^-6..2^-10}, 100k OOS | feb6c7b / de1ddfc ‡ | — | log–log 斜率 0.7129 ± 0.0017，R² ≈ 0.999 | ✅ |
| e4h (A2) | seeds 0–4, Heston d=2, λ=1, ε∈{2^-6..2^-10}, 100k OOS | feb6c7b / de1ddfc ‡ | — | log–log 斜率 0.6469 ± 0.0173（中位 0.6545） | ✅ |

\* `git_commit: c3bc2ab` 是起跑时的 HEAD；该批次 5 份 `metrics.json` 记 `dirty: true`，因为文档改动当时尚未提交
（`git status` 只有 `README.md` 与 `docs/`，`src/` 零改动）。重跑时 `OMP_NUM_THREADS=2`，单 seed 32.5 min。溯源见 [changelog](changelog.md) §3。
† B1 提交于分支 `feat/b1-hk-data-calib`，并在干净工作区下重跑 `python -m src.hk.calibrate`（`results/calib/{hsi,hsce}_garch.json` 均记 `"dirty": false`）。
‡ A2 逐 seed 明细、源分支 commit 与 provenance 说明见 [docs/A2/experiments-e2-e4.md](A2/experiments-e2-e4.md)。
