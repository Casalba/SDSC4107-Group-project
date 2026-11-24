from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.common.config import dump_runtime_config, load_config, resolve_seeds
from src.common.logging import human_secs, setup_logger
from src.common.metrics import capital_requirement, save_pnl, write_metrics
from src.common.seed import resolve_device, seed_everything
from src.common.types import RunPaths
from src.hk.synth import HKMarket, HKWorld
from src.replication.losses import avg_cvar
from src.replication.net_recurrent import SemiRecurrentPolicy


def build_m2_policy(cfg: dict[str, Any], device: torch.device | str = "cpu") -> SemiRecurrentPolicy:
    pol_cfg = cfg.get("policy", {})
    n_steps = int(cfg.get("n_steps", 30))
    policy = SemiRecurrentPolicy(
        n_features=2,
        d=1,
        n_steps=n_steps,
        hidden=int(pol_cfg.get("hidden", 16)),
        recurrent=bool(pol_cfg.get("recurrent", True)),
        batch_norm=bool(pol_cfg.get("batch_norm", True)),
        norm=pol_cfg.get("norm", "layer"),
        out_scale=pol_cfg.get("out_scale", 1.0),
    )
    return policy.to(device=device, dtype=torch.float64)


def train_m2_single_seed(
    cfg: dict[str, Any],
    seed: int,
    device: torch.device,
    log: logging.Logger,
    mode: str = "garch",
    index: str = "hsi",
) -> dict[str, Any]:
    seed_everything(seed)
    n_steps = int(cfg.get("n_steps", 30))
    batch_size = int(cfg.get("batch", 1024))
    eps = float(cfg.get("eps", 0.0))
    alpha = float(cfg.get("alpha", 0.5))
    market_cfg = cfg.get("market", {})

    hk_market = HKMarket(
        index=index,
        mode=mode,
        s0=float(market_cfg.get("s0", 100.0)),
        strike=float(market_cfg.get("k", 100.0)),
        maturity=float(market_cfg.get("t", 30.0 / 365.0)),
        n_steps=n_steps,
        eps=eps,
        d=1,
    )
    world = HKWorld(hk_market, device=device, dtype=torch.float64)

    train_cfg = cfg.get("train", {})
    n_iter = int(train_cfg.get("n_iter", 30000))
    lr = float(train_cfg.get("lr", 0.005))
    lr_min = float(train_cfg.get("lr_min", 5.0e-4))
    lr_schedule = str(train_cfg.get("lr_schedule", "cosine"))
    eval_paths = int(train_cfg.get("eval_paths", 100000))
    eval_seed_offset = int(train_cfg.get("eval_seed_offset", 10000))
    log_every = int(train_cfg.get("log_every", 2500))

    policy = build_m2_policy(cfg, device=device)
    policy.train()

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(n_iter, 1), eta_min=lr_min)
        if lr_schedule == "cosine"
        else None
    )

    for step in range(1, n_iter + 1):
        optimizer.zero_grad(set_to_none=True)
        paths = world.simulate(batch_size)
        feats = world.features(paths)
        positions = policy.forward_path(feats)  # (B, n_steps, 1)
        loss_vec = world.terminal_loss(positions, paths)
        obj = avg_cvar(loss_vec, alpha=alpha)
        obj.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if log_every > 0 and (step % log_every == 0 or step == 1 or step == n_iter):
            log.info(
                "iter=%d/%d obj_cvar(%.2f)=%.5f lr=%.2e",
                step,
                n_iter,
                alpha,
                float(obj.item()),
                optimizer.param_groups[0]["lr"],
            )

    policy.eval()
    seed_everything(seed + eval_seed_offset)
    with torch.no_grad():
        eval_paths_obj = world.simulate(eval_paths)
        eval_feats = world.features(eval_paths_obj)
        eval_pos = policy.forward_path(eval_feats)
        eval_loss = world.terminal_loss(eval_pos, eval_paths_obj)
        eval_cost = world.cost(eval_pos, eval_paths_obj)

        losses_np = eval_loss.cpu().numpy()
        cap = capital_requirement(eval_loss)
        obj_val = float(avg_cvar(eval_loss, alpha=alpha).item())

    metrics_out = {
        "train_objective": obj_val,
        "c95": float(cap["c95"]),
        "c99": float(cap["c99"]),
        "mean_loss": float(np.mean(losses_np)),
        "mean_cost": float(eval_cost.mean().item()),
    }
    checks = {
        "finite_loss": bool(np.isfinite(losses_np).all()),
        "positive_c95": bool(metrics_out["c95"] > 0.0),
    }

    pol_cfg = cfg.get("policy", {})
    ckpt_payload = {
        "state_dict": {k: v.cpu() for k, v in policy.state_dict().items()},
        "meta": {
            "policy": "policy_hk_m2",
            "seed": int(seed),
            "index": index,
            "mode": mode,
            "n_features": 2,
            "d": 1,
            "n_steps": n_steps,
            "hidden": int(pol_cfg.get("hidden", 16)),
            "recurrent": bool(pol_cfg.get("recurrent", True)),
            "batch_norm": bool(pol_cfg.get("batch_norm", True)),
            "norm": pol_cfg.get("norm", "layer"),
            "out_scale": pol_cfg.get("out_scale", 1.0),
            "eps": eps,
            "alpha": alpha,
            "dtype": "float64",
        },
    }

    return {
        "metrics": metrics_out,
        "checks": checks,
        "n_paths": eval_paths,
        "terminal_loss": losses_np,
        "extra_pnl": {
            "positions": eval_pos.squeeze(-1).cpu().numpy(),
            "cost": eval_cost.cpu().numpy(),
        },
        "artifacts": {"policy.pt": ckpt_payload},
    }


def load_m2_policy(
    ckpt_path: str | Path, device: torch.device | str = "cpu"
) -> SemiRecurrentPolicy:
    p = Path(ckpt_path)
    if not p.exists():
        raise FileNotFoundError(f"M2 checkpoint not found: {p}")
    payload = torch.load(p, map_location="cpu", weights_only=True)
    meta = payload["meta"]
    policy = SemiRecurrentPolicy(
        n_features=int(meta["n_features"]),
        d=int(meta["d"]),
        n_steps=int(meta["n_steps"]),
        hidden=int(meta["hidden"]),
        recurrent=bool(meta["recurrent"]),
        batch_norm=bool(meta.get("batch_norm", True)),
        norm=meta.get("norm", "layer"),
        out_scale=meta.get("out_scale", 1.0),
    )
    policy.load_state_dict(payload["state_dict"])
    policy.to(device=device, dtype=torch.float64).eval()
    for param in policy.parameters():
        param.requires_grad_(False)
    return policy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HK M2 Local Re-fit Trainer (B2)")
    parser.add_argument("--config", required=True, help="港股配置檔路徑")
    parser.add_argument("--mode", choices=["garch", "garch_plain", "bootstrap"], default="garch")
    parser.add_argument("--index", choices=["hsi", "hsce"], default="hsi")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-iter", type=int, default=None)
    parser.add_argument("--out", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    if args.out is not None:
        cfg["out_root"] = str(args.out)
    if args.alpha is not None:
        cfg["alpha"] = float(args.alpha)
    if args.eps is not None:
        cfg["eps"] = float(args.eps)
    if args.n_iter is not None:
        cfg.setdefault("train", {})["n_iter"] = int(args.n_iter)

    seeds = resolve_seeds(cfg, args.seed)
    device = resolve_device(cfg.get("device", "auto"))

    kind_tag = str(cfg.get("kind", "hk_m2"))
    alpha_tag = f"a{int(round(float(cfg['alpha']) * 100)):02d}"
    run_group = f"{kind_tag}_{args.index}_{args.mode}_m2_{alpha_tag}"

    for seed in seeds:
        started = time.perf_counter()
        paths = RunPaths(Path(str(cfg["out_root"])), run_group, seed).mkdir()
        log = setup_logger(f"dh.{run_group}.{seed}", paths.log)
        dump_runtime_config(cfg, paths.config)

        payload = train_m2_single_seed(
            cfg=cfg, seed=seed, device=device, log=log, mode=args.mode, index=args.index
        )
        runtime = time.perf_counter() - started
        write_metrics(
            paths.metrics,
            exp=str(cfg["exp"]),
            seed=seed,
            cfg=cfg,
            metrics=payload["metrics"],
            checks=payload["checks"],
            runtime_sec=runtime,
            device=device,
            n_paths=int(payload["n_paths"]),
        )
        save_pnl(paths.pnl, payload["terminal_loss"], **payload.get("extra_pnl", {}))
        for name, obj in payload.get("artifacts", {}).items():
            torch.save(obj, paths.run_dir / name)

        print(
            f"[{run_group}/{seed}] c95={payload['metrics']['c95']:.4f} "
            f"c99={payload['metrics']['c99']:.4f} runtime={human_secs(runtime)} -> {paths.run_dir}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
