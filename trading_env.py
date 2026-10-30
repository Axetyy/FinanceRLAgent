import numpy as np
import pandas as  pd
import gymnasium as gym
from gymnasium import spaces

import config as C
from download_data import FEATURE_COLS

class TradingEnv(gym.Env):
    metadata = {"render_modes":[]}
    
    def __init__(self, panel: pd.DataFrame, tickers=None, news=None,
                 window=C.WINDOW, fee_bps=C.FEE_BPS, slippage_bps=C.SLIPPAGE_BPS,
                 impact_coef=C.IMPACT_COEF, exec_delay=C.EXEC_DELAY,
                 long_only=C.LONG_ONLY, logit_scale=C.LOGIT_SCALE,
                 deadband=C.DEADBAND, churn_penalty=C.CHURN_PENALTY,
                 dd_penalty=C.DD_PENALTY, reward_scale=C.REWARD_SCALE,
                 aux_horizons=C.AUX_HORIZONS, episode_len=None):
        super().__init__()
        self.tickers = tickers or list(panel.columns.get_level_values(0).unique())
        self.n=  len(self.tickers)
        self.window = window
        self.fee = fee_bps / 1e4 
        self.slip = slippage_bps / 1e4
        self.impact =impact_coef / 1e4
        self.delay = exec_delay
        self.long_only = long_only
        self.logit_scale = logit_scale
        self.deadband = deadband
        self.churn = churn_penalty
        self.dd_pen = dd_penalty
        self.scale = reward_scale
        self.horizons = tuple(aux_horizons)
        self.episode_len = episode_len
        
        self.feats = np.stack([panel[(t,f)].to_numpy() for t in self.tickers for f in FEATURE_COLS ],axis=1).astype(np.float32)
        self.prices = np.stack([panel[(t,"Close")].to_numpy() for t in self.tickers],axis=1)
        
        self.dates = panel.index
        self.T = len(panel)
        self.nf = self.feats.shape[1]
        self.news = None
        if news is not None:
            news = np.asarray(news, dtype=np.float32)
            assert news.shape[:2] == (self.T, self.n), "news must be (T, n_assets, D)"
            self.news = news
            has = np.abs(news).sum(axis=2) > 0                           # (T, n)
            idx = np.where(has, np.arange(self.T)[:, None], -1)
            self.news_last = np.maximum.accumulate(idx, axis=0)          # last news index <= t
            self.news_dim = news.shape[2]
            
            
        n_act = self.n + 1 if long_only else self.n
        self.action_space = spaces.Box(-1.0,1.0,shape=(n_act,),dtype=np.float32)
        obs = {"market":spaces.Box(-10.0,10.0,shape = (self.window,self.nf),dtype=np.float32),
               "portfolio":spaces.Box(-2.0,2.0,shape=(3*self.n+2,),dtype=np.float32)
        }
        if self.news is not None:
            obs["news"] = spaces.Box(-1.0, 1.0, shape=(self.n, self.news_dim), dtype=np.float32)
            obs["news_age"] = spaces.Box(0.0, 1.0, shape=(self.n,), dtype=np.float32)
        self.observation_space = spaces.Dict(obs)
        
    def _to_weights(self,action):
        a = np.clip(np.asarray(action))        
        if self.long_only:
            z = a * self.logit_scale
            e = np.exp(z- z.max())
            return (e/ e.sum())[:-1]
        gross = np.abs(a).sum()
        return a / gross if gross > 1 else a # cap gross at 1
    
    def _obs(self):
        lo = self.t - self.window + 1
        px = self.prices[self.t]
        held = np.abs(self.w) > 1e-3
        missing = held & np.isnan(self.entry)
        self.entry[missing] = px[missing]
        self.age[~held] = 0
        unreal = np.zeros(self.n)
        unreal[held] = np.sign(self.w[held]) * (px[held] / self.entry[held] - 1.0)
        port = np.concatenate([
            self.w,
            np.clip(unreal, -1, 1),
            np.clip(self.age / C.AGE_CAP, 0, 1),
            [self.dd, 1.0 - np.abs(self.w).sum()],
        ]).astype(np.float32)
        obs = {"market": self.feats[lo:self.t + 1].copy(), "portfolio": port}
        if self.news is not None:
            last = self.news_last[self.t]                               # (n,)
            emb = np.zeros((self.n, self.news_dim), np.float32)
            age = np.ones(self.n, np.float32)
            ok = last >= 0
            emb[ok] = self.news[last[ok], np.arange(self.n)[ok]]
            age[ok] = np.clip((self.t - last[ok]) / C.NEWS_AGE_CAP, 0, 1)
            obs["news"], obs["news_age"] = emb, age
        return obs
    def _fill(self,target):
        """Rebalance toward `target` at the current close with a per-asset deadband.
        Returns (cost as fraction of equity, executed turnover)."""
        
        delta = target - self.w
        delta = np.where(np.abs(delta)<self.deadband,0.0,delta)
        old,self.w = self.w,self.w+delta
        turnover = float(np.abs(delta).sum())
        cost = turnover * (self.fee + self.slip) + self.impact * turnover ** 2
        px = self.prices[self.t]
        for i in range(self.n):
            if abs(self.w[i]) < 1e-3:
                self.entry[i],self.age[i] = np.nan,0
            elif abs(old[i]) < 1e-3 or np.sign(old[i]) != np.sign(self.w[i]):
                self.entry[i],self.age[i] = px[i],0
        return cost,turnover
    def _aux(self):
        tgt = np.zeros((self.n,len(self.horizons)),np.float32)
        mask = np.zeros(len(self.horizons),np.float32)
        for j,h in enumerate(self.horizons):
            if self.t + h < self.T:
                tgt[:, j] = self.prices[self.t + h] / self.prices[self.t] - 1.0
                mask[j] = 1.0
        return tgt,mask
    
    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        lo = self.window - 1
        max_start = self.T - 2 -(self.episode_len or 0)
        self.t = lo if self.episode_len is None else int(
            self.np_random.integers(lo,max(lo+1,max_start)))
        self.t0 = self.t
        self.w = np.zeros(self.n)
        self.entry = np.full(self.n,np.nan)
        self.age = np.zeros(self.n)
        self.equity = self.peak = 1.0
        self.dd = 0.0
        self.queue = []
        self.curve = [self.equity]
        return self._obs(),{}
    
    def step(self,action):
        target = self._to_weights(action)
        cost = turnover = 0.0
        
        if self.delay == 0:
            c,tv = self._fill(target)
            cost += c
            turnover += tv
        else:
            self.queue.append((self.t + self.delay,target))
        
        r = self.prices[self.t + 1] / self.prices[self.t] - 1.0
        port_ret = float(self.w @ r) 
        self.w = self.w *(1+r) / (1 + port_ret) # weight drifting
        self.t += 1
        self.age[np.abs(self.w) > 1e-3] +=1
        
        due =[q for q in self.queue if q[0] <= self.t]
        self.queue = [q for q in self.queue if q[0] > self.t]
        for _ ,tgt in due:
            c,tv = self._fill(tgt)
            cost += c
            turnover += tv
        
        step_ret  = port_ret - cost
        self.equity*=(1+step_ret)
        self.curve.append(self.equity)
        
        self.peak = max(self.peak,self.equity)
        new_dd = 1.0 - self.equity / self.peak
        dd_increase = max(0.0,new_dd - self.dd)
        self.dd = new_dd
        
        reward = self.scale * (np.log1p(max(step_ret, -0.99)) - self.dd_pen * dd_increase - self.churn * turnover)
        
        terminated = self.t >= self.T -2
        truncated = (self.episode_len is not None and (self.t - self.t0) >=self.episode_len)
        
        aux,aux_mask = self._aux()
        info = {"equity": self.equity, "cost": cost, "turnover": turnover,
                "weights": self.w.copy(), "date": self.dates[self.t],
                "aux_target": aux, "aux_mask": aux_mask}
        return self._obs(),float(reward),terminated,truncated,info
    
def make_env(split:str,news=None,**kw):
    panel = pd.read_parquet(f"{C.DATA_DIR}/{split}.parquet")
    return TradingEnv(panel,news=news,**kw)
        
        
            