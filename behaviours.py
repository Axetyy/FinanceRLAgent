"""Behaviour policies used to generate the OFFLINE dataset for the Decision Transformer.
 
A Decision Transformer learns "what action goes with this state AND this achieved
return", so the dataset must contain a SPREAD of trajectory quality: good, mediocre
and bad. We therefore mix momentum rules, random holders, cash, buy&hold and a
hindsight "oracle" (peeks at future prices - allowed ONLY here, to create high-return
examples; the model never sees the future at inference), each with random hold times,
random-action probability and action noise.
"""

import numpy as np

class Behaviour:
    def __init__(self,env,kind,rng,hold,eps,noise,lookback,horizon,topk):
        assert env.long_only, "behaviours assume the softmax-with-cash action space"
        self.env,self.kind,self.rng = env,kind,rng
        self.hold,self.eps,self.noise = hold,eps,noise
        self.lookback,self.horizon,self.topk = lookback,horizon,topk
        self.cur = None
        self.next_decide = 0
    def _select(self,score):
        sel = score > 0
        if self.topk > 0 and sel.sum() > self.topk:
            top = np.zeros(self.env.n,bool)
            top[np.argsort(-score)[:self.topk]]=True
            sel &=top
        return sel
    
    def _decide(self, t):
        env, n = self.env, self.env.n
        if self.rng.random() < self.eps or self.kind == "random":
            sel = self.rng.random(n) < 0.5
        elif self.kind == "buyhold":
            sel = np.ones(n, bool)
        elif self.kind == "cash":
            sel = np.zeros(n, bool)
        elif self.kind == "momentum":
            score = env.prices[t] / env.prices[max(t - self.lookback, 0)] - 1.0
            sel = self._select(score)
        else:  # "oracle": hindsight
            score = env.prices[min(t + self.horizon, env.T - 1)] / env.prices[t] - 1.0
            sel = self._select(score)
        a = -np.ones(n + 1, np.float32)
        a[:n][sel] = 1.0
        if not sel.any():
            a[n] = 1.0
        return a
    
    def __call__(self):
        t = self.env.t
        if self.cur is None or t >= self.next_decide:
            self.cur = self._decide(t)
            self.next_decide = t + self.hold
        a = self.cur
        if self.noise > 0:
            a = np.clip(a + self.rng.normal(0, self.noise, a.shape), -1, 1)
        return a.astype(np.float32)
    
def make_random_behavior(env, rng):
    kind = str(rng.choice(["buyhold", "momentum", "oracle", "random", "cash"],
                          p=[0.05, 0.30, 0.35, 0.20, 0.10]))
    return Behaviour(
        env, kind, rng,
        hold=int(rng.choice([1, 5, 10, 20, 40])),
        eps=float(rng.choice([0.0, 0.1, 0.3, 0.6])),
        noise=float(rng.choice([0.0, 0.05, 0.2])),
        lookback=int(rng.choice([10, 20, 40, 60])),
        horizon=int(rng.choice([5, 10, 20])),
        topk=int(rng.choice([0, 1, 2, 3])),
    )