import torch.nn as nn
import torch
import torch.nn.functional as F
from model.modules import Block,RMSNorm,build_RoPE

class DecisionTransformerPolicy(nn.Module):
    def __init__(self,n_assets,n_feat,window,port_dim,act_dim,n_aux_h,news_dim=0,d_model=128,n_layers=4,n_heads=4,n_kv_heads=2,dropout=0.0,attn_window=None,init_log_std = -1.0):
        super().__init__()
        self.cfg = dict(n_assets=n_assets, n_feat=n_feat, window=window,
                        port_dim=port_dim, act_dim=act_dim, n_aux_h=n_aux_h,
                        news_dim=news_dim, d_model=d_model, n_layers=n_layers,
                        n_heads=n_heads, n_kv_heads=n_kv_heads, dropout=dropout,
                        attn_window=attn_window, init_log_std=init_log_std)
        
        self.n_assets, self.n_aux_h, self.news_dim = n_assets, n_aux_h, news_dim
        self.market_in = nn.Linear(n_feat, d_model)
        self.rho_in = nn.Sequential(
            nn.Linear(1, d_model), 
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        self.port_in =nn.Linear(port_dim,d_model)
        if news_dim > 0:
            self.news_in = nn.Linear(news_dim, d_model)
            self.asset_emb = nn.Embedding(n_assets, d_model)
            self.age_in = nn.Linear(1, d_model)
            
        self.n_prefix  = 2 + (n_assets if news_dim > 0 else 0)
        self.seq_len  = self.n_prefix + window
        
        self.blocks = nn.ModuleList([
            Block(d_model,n_heads,n_kv_heads,dropout) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        
        self.mu_head = nn.Linear(d_model,act_dim)
        self.value_head = nn.Sequential(
        nn.Linear(d_model,d_model),
        nn.SiLU(),
        nn.Linear(d_model,1),
        )
        self.aux_head = nn.Linear(d_model,n_assets * n_aux_h)
        self.log_std = nn.Parameter(torch.full((act_dim,),float(init_log_std)))
        
        
        self.register_buffer("aux_scale", torch.ones(n_aux_h))
        
        cos,sin = build_RoPE(d_model // n_heads,self.seq_len)
        self.register_buffer("rope_cos",cos,persistent=False)
        self.register_buffer("rope_sin",sin,persistent=False)
        
        mask = None
        if attn_window is not None:
            S = self.seq_len
            i = torch.arange(S)[:,None]
            j = torch.arange(S)[None,:]
            causal = j <= i
            local = (i-j) < attn_window
            glob = j - self.n_prefix
            
            mask = causal & (local | glob)
        self.register_buffer("attn_mask",mask,persistent=False)
        with torch.no_grad():
            self.mu_head.weight.mul_(0.01)
            self.mu_head.bias.zero_()
            
    def features(self,obs,rho):
        B =obs['market'].shape[0]
        parts = [self.rho_in(rho.view(B,1).float()).unsqueeze(1),self.port_in(obs['portfolio']).unsqueeze(1)]
        if self.news_dim > 0:
            if "news" not in obs:
                raise ValueError("model was built with news but obs has no 'news'")
            ids = torch.arange(self.n_assets, device=obs["news"].device)
            parts.append(self.news_in(obs["news"]) + self.asset_emb(ids)[None]
                         + self.age_in(obs["news_age"].unsqueeze(-1)))
        parts.append(self.market_in(obs["market"]))
        dt = parts[-1].dtype
        x = torch.cat([p.to(dt) for p in parts],dim=1)
        assert x.shape[1] == self.seq_len, f"seq {x.shape[1]} != {self.seq_len}"
        for blk in self.blocks:
            x = blk(x, self.rope_cos, self.rope_sin, self.attn_mask)
        return self.norm(x)[:,-1]
    
    def forward(self,obs,rho,detach_value=False):
        h  = self.features(obs,rho)
        mu = torch.tanh(self.mu_head(h)).float()
        hv = h.detach() if detach_value else h
        value = self.value_head(hv).squeeze(-1).float()
        aux = self.aux_head(h).float().view(-1,self.n_assets,self.n_aux_h)
        return mu,value,aux
    
    def get_action_and_value(self,obs,rho,action=None,deterministic=False,detach_value=False):
        mu,value,aux = self.forward(obs,rho,detach_value)
        std =self.log_std.clamp(-3.0,0.5).exp().expand_as(mu)
        dist = torch.distributions.Normal(mu,std)
        if action is None:
            action = mu if deterministic else dist.sample()
        logp = dist.log_prob(action).sum(-1)
        ent = dist.entropy().sum(-1)
        return action,logp,ent,value,mu,aux
    
def load_policy(path,device):
    ck = torch.load(path,map_location=device,weights_only=False)
    model = DecisionTransformerPolicy(**ck['cfg']).to(device)
    model.load_state_dict(ck["model"])
    return model,ck["meta"]

def save_policy(path,model,meta):
    torch.save({"model": model.state_dict(), "cfg": model.cfg, "meta": meta}, path)
        
