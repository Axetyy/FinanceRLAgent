import numpy as np
import pandas as pd
import torch
import config as C

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
def amp_ctx(device, enabled):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=bool(enabled) and device.type == "cuda")
def load_panel(split):
    return pd.read_parquet(f"{C.DATA_DIR}/{split}.parquet")

def load_news(path):
    """News embeddings aligned to a split: float32 array (T, n_assets, D).
    All-zero rows mean 'no news at that step'."""
    return None if not path else np.load(path).astype(np.float32)

def stack_obs(obs_list):
    return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}

def to_torch(obs, device):
    return {k: torch.as_tensor(v, dtype=torch.float32, device=device)
            for k, v in obs.items()}
class StaticTensors:
    def __init__(self, env, device):
        self.device = device
        self.W, self.T, self.n = env.window, env.T, env.n
        feats = torch.from_numpy(env.feats).to(device)
        self.windows = feats.unfold(0, self.W, 1)
 
        H = len(env.horizons)
        aux = np.zeros((env.T, env.n, H), np.float32)
        mask = np.zeros((env.T, H), np.float32)
        for j, h in enumerate(env.horizons):
            if h < env.T:
                aux[:env.T - h, :, j] = env.prices[h:] / env.prices[:env.T - h] - 1.0
                mask[:env.T - h, j] = 1.0
        self.aux = torch.from_numpy(aux).to(device)
        self.aux_mask = torch.from_numpy(mask).to(device)
 
        self.news = self.news_age = None
        if env.news is not None:
            T, n, D = env.news.shape
            # (T, n), -1 = none yet
            last = env.news_last
            ok = last >= 0
            emb = np.zeros((T, n, D), np.float32)
            age = np.ones((T, n), np.float32)
            cols = np.broadcast_to(np.arange(n), (T, n))
            rows = np.broadcast_to(np.arange(T)[:, None], (T, n))
            emb[ok] = env.news[last[ok], cols[ok]]
            age[ok] = np.clip((rows[ok] - last[ok]) / C.NEWS_AGE_CAP, 0, 1)
            self.news = torch.from_numpy(emb).to(device)
            self.news_age = torch.from_numpy(age).to(device)
 
    def batch(self, ts):
        obs = {"market": self.windows[ts - (self.W - 1)].permute(0, 2, 1).contiguous()}
        if self.news is not None:
            obs["news"] = self.news[ts]
            obs["news_age"] = self.news_age[ts]
        return obs
 
    def aux_batch(self, ts):
        return self.aux[ts], self.aux_mask[ts]
    
    
def aux_loss_fn(aux_pred,aux_target,aux_mask,aux_scale):
    """Masked MSE """
    target = (aux_target / aux_scale.view(1,1,-1)).clamp(-5,5)
    m = aux_mask.unsqueeze(1)
    n = aux_pred.shape[1    ]
    return (((aux_pred-target)**2)*m).sum() / (aux_mask.sum()*n + 1e-8)

def metrics(curve):
    c = np.asarray(curve,dtype=np.float64)
    r =  c[1:] / c[:-1] - 1.0
    return {"total_return": float(c[-1] / c[0] - 1.0),
            "sharpe": float(r.mean() / (r.std() + 1e-12) * np.sqrt(252)),
            "max_dd": float((1.0 - c / np.maximum.accumulate(c)).max())}
    
@torch.no_grad()
def run_episode(policy,env,rho_z,device):
    policy.eval()
    obs,_ = env.reset(seed = 0)
    rho = torch.full((1,),float(rho_z),device=device)   
    done = False
    turnover = cost = 0.0
    cash = []
    while not done:
        o = to_torch(stack_obs([obs]), device)
        action = policy.get_action_and_value(o, rho, deterministic=True)[0]
        obs, _, term, trunc, info = env.step(action[0].cpu().numpy())
        done = term or trunc
        turnover += info["turnover"]
        cost += info["cost"]
        cash.append(1.0 - float(np.abs(info["weights"]).sum()))
    stats = {"turnover": turnover, "cost": cost, "avg_cash": float(np.mean(cash))}
    return env.curve, stats
    
    
def evaluate_policy(policy, env, rho_z, device, verbose=True, label=""):
    t0 = env.window - 1
    curve, stats = run_episode(policy, env, rho_z, device)
    t_end = env.t
    p = env.prices[t0:t_end + 1]
    curves = {"Agent": np.asarray(curve)}
    if "SPY" in env.tickers:
        i = env.tickers.index("SPY")
        curves["SPY buy&hold"] = p[:, i] / p[0, i]
    curves["Equal-weight B&H"] = (p / p[0]).mean(axis=1)
    res = {k: metrics(v) for k, v in curves.items()}
    if verbose:
        d0, d1 = env.dates[t0].date(), env.dates[t_end].date()
        print(f"\n== {label} {d0} -> {d1} ({t_end - t0} steps) ==")
        print(f"  {'':18s}{'return':>9s}{'sharpe':>8s}{'maxDD':>8s}")
        for k, m in res.items():
            print(f"  {k:18s}{m['total_return']:9.1%}{m['sharpe']:8.2f}{m['max_dd']:8.1%}")
        print(f"  agent: avg cash {stats['avg_cash']:.0%} | total turnover "
              f"{stats['turnover']:.1f} | total costs {stats['cost']:.4f}")
    return {"metrics": res, "curves": curves, "stats": stats}