# SDSC4107 · Deep Hedging 复现 × 港股交叉验证

> 状态：**G0 骨架就绪**（`src/common`、`src/policy`、`train` 入口、测试与 CI）。下一步：pfhedge 适配层（`env_heston.py` / `env_bs.py` / `net_recurrent.py` / `losses.py`）。

## 先读两份文档

| 文件 | 管什么 |
|---|---|
| [docs/plan.md](docs/plan.md) | 做什么、做到什么算对：E1–E4 复现靶子、港股 C\* 口径、判据与风险 |
| [docs/spec.md](docs/spec.md) | 谁做什么、交到哪、怎么命名、怎么验收：角色边界、目录命名、接口 schema、每人 DoD、Gate |

冲突时：范围与判据看 plan，格式与流程看 spec。

## 目录

```
project/
├─ README.md            # 本文件
├─ pyproject.toml       # pytest / ruff 配置
├─ requirements.lock.txt
├─ .venv/               # Python 3.12 环境（828 MB，已从上级目录迁入）
├─ docs/
│   ├─ plan.md          # 方案 v1.3
│   ├─ spec.md          # 协作规范 v1.2
│   ├─ experiments.md   # 实验登记表（spec §9）
│   └─ paper/           # deephedging.pdf + section5.txt / section5b.txt（论文与摘录）
├─ src/
│   ├─ common/          # config seed logging metrics types
│   ├─ policy/          # policy.py（冻结接口）
│   └─ replication/     # train.py + smoke.py（G0）；env_* / losses / benchmark 待补
├─ configs/exp/         # debug.yaml（G0）；e1.yaml ... 待补
├─ tests/
├─ scripts/             # ci.py；fetch_hk.py / check_b0.py / repro.py 待补
├─ figures/             # 报告终版图
├─ reports/             # 英文正文
└─ refs/                # 第三方仓库快照，只读参考，不改、不提交
    ├─ pfhedge/         # pfnet-research/pfhedge @ 1fc08c7 (2024-08-30)
    └─ deephedging/     # hansbuehler/deephedging @ fe53fd9 (2026-01-22)（notebooks 保留）
```

`data/`、`results/` 运行时生成、不进 git（spec §4 / §7）。

## 跑起来（5 行）

```powershell
cd E:\cityu\sdsc4107\project
.venv\Scripts\python.exe -m pip install -r requirements.lock.txt   # 首次装环境
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m src.replication.train --config configs/exp/debug.yaml
# -> results/debug/0/{metrics.json, config.yaml, run.log, pnl.npz}
```

本地 CI（ruff + pytest，与 GitHub Actions 同源）：`.venv\Scripts\python.exe scripts\ci.py`。

## 参考仓库怎么看

- `refs/pfhedge/`：我们要用的库本体。看 `pfhedge/nn/modules/`（风险测度）、`pfhedge/instruments/`（HestonStock / VarianceSwap）、`examples/`（训练循环写法）。本地已装 0.23.0 于 `project/.venv`。
- `refs/deephedging/`：论文原作者的 TensorFlow 实现，看 `objectives.py`（风险测度）、`layers.py`（网络结构）、`world.py`（Heston 世界）。68 MB 里大部分是 `notebooks/*.ipynb`，源码只有 ~10 个 `.py`。

## 环境

- Python 环境就在本目录：`project\.venv`（828 MB，已从上级 `.venv` 迁入，console 脚本已重生成）。
  - 直接调用：`.venv\Scripts\python.exe -m pip ...`；`pip` / `pip3` / `pip3.12` 已修复可用。
  - 已装：pfhedge 0.23.0、torch 2.15.0+cpu、numpy 1.26.4、pandas 3.0.6、matplotlib 3.11.2、sympy 1.14.0。
- **torch 目前是 CPU 版，这是刻意的**：组里四台机器不统一，代码用 `device=auto`，CPU 也能跑通全流程。有 N 卡的（我这台 5070 Ti 属 sm_120，要 cu128 版）各自按机器装；每个结果里的 `device` / `torch_version` 会记录是哪台机器跑的（spec §5.2）。
- 数据不进 git：`data/`、`results/` 都忽略，抓取脚本在 `scripts/`（开工后建）。
