"""Evaluate a checkpoint on val or test and compare with SPY / equal-weight buy&hold.

    python evaluate.py --ckpt checkpoints/dt_best.pt --split val --sweep
    python evaluate.py --ckpt checkpoints/ppo_best.pt --split val --plot
    python evaluate.py --ckpt checkpoints/ppo_best.pt --split test     # use ONCE, at the very end

--sweep : ask the Decision Transformer for different target returns (rho quantiles)
          to see whether conditioning actually changes behaviour/returns.
"""
import argparse
import os

import config as C
from trading_env import TradingEnv
from model.model import load_policy
from common import get_device, load_panel, load_news, evaluate_policy


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--news", default=None)
    p.add_argument("--rho", type=float, default=None)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--plot", action="store_true")
    return p.parse_args()


def main():
    args = parse()
    device = get_device()
    policy, meta = load_policy(args.ckpt, device)
    env = TradingEnv(load_panel(args.split), news=load_news(args.news))
    mean, std = meta["rho_mean"], meta["rho_std"]

    if args.sweep:
        targets = {f"q{q}": v for q, v in meta["rho_quantiles"].items()}
    else:
        targets = {"target": args.rho if args.rho is not None else meta["rho_target"]}

    last = None
    for name, rho in targets.items():
        z = (rho - mean) / (std + 1e-9)
        last = evaluate_policy(policy, env, z, device, label=f"{args.split} rho={name} ({rho:.3f})")

    if args.plot and last is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            os.makedirs("reports", exist_ok=True)
            plt.figure(figsize=(10, 5))
            for k, v in last["curves"].items():
                plt.plot(v, label=k)
            plt.legend()
            plt.title(f"{args.split}: equity curves (start = 1.0)")
            plt.grid(alpha=0.3)
            out = f"reports/{args.split}_equity.png"
            plt.savefig(out, dpi=130, bbox_inches="tight")
            print("saved", out)
        except ImportError:
            print("matplotlib not installed - pip install matplotlib to enable --plot")


if __name__ == "__main__":
    main()