import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self,dim,eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self,x):
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1,keepdim=True) + self.eps)
        return ( out * self.weight).to(x.dtype)
    
def build_RoPE(head_dim,max_len,base = 10000.0):
    inv = 1.0 / (base ** (torch.arange(0,head_dim,2).float()/head_dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t,inv)
    return freqs.cos(), freqs.sin()

def apply_RoPE(x,cos,sin):
    """ x: (B,H,S,head_dim). """
    S = x.shape[2]
    cos = cos[:S].to(x.dtype)
    sin = sin[:S].to(x.dtype)
    x1,x2 = x[...,0::2],x[...,1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    return torch.stack((o1,o2),dim=-1).flatten(-2)

class Attention(nn.Module):
    def __init__(self, d_model,n_heads,n_kv_heads,dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0 and n_heads % n_kv_heads == 0
        self.h,self.hkv = n_heads,n_kv_heads
        self.hd = d_model // n_heads
        assert self.hd % 2 == 0, "head_dim must be even for RoPE"
        self.q = nn.Linear(d_model,self.h * self.hd, bias=False)
        self.k = nn.Linear(d_model,self.hkv * self.hd,bias=False)
        self.v = nn.Linear(d_model,self.hkv * self.hd,bias=False)
        self.o = nn.Linear(self.h * self.hd,d_model,bias=False)
        self.qn = RMSNorm(self.hd)
        self.kn = RMSNorm(self.hd)
        self.dropout = dropout
        
    def forward(self,x,cos,sin,mask):
        B,S,_ = x.shape
        q = self.q(x).view(B,S,self.h,self.hd).transpose(1,2)
        k = self.k(x).view(B,S,self.hkv,self.hd).transpose(1,2)
        v = self.v(x).view(B,S,self.hkv,self.hd).transpose(1,2)
        
        q,k = self.qn(q),self.kn(k)
        q,k = apply_RoPE(q,cos,sin),apply_RoPE(k,cos,sin)
        if self.hkv != self.h:
            rep = self.h // self.hkv
            k = k.repeat_interleave(rep,dim=1)
            v = v.repeat_interleave(rep,dim=1)
        p = self.dropout if self.training else 0.0
        if mask is None:
            out = F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=p)
        else:
            out = F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=p)
        out = out.transpose(1,2).reshape(B,S,self.h * self.hd)
        return self.o(out)
    
class SwiGLU(nn.Module):
    def __init__(self,d_model,mult = 4):
        super().__init__()
        hidden = int(2* d_model * mult / 3)
        hidden = (hidden + 7) // 8 * 8
        self.w1 = nn.Linear(d_model,hidden,bias=False)
        self.w2 = nn.Linear(d_model,hidden,bias=False)
        self.w3 = nn.Linear(hidden,d_model,bias=False)
        
    def forward(self,x):
        return self.w3(F.silu(self.w1(x))*self.w2(x))
    
class Block(nn.Module):
    def __init__(self,d_model,n_heads,n_kv_heads,dropout):
        super().__init__()
        self.n1 = RMSNorm(d_model)
        self.attn = Attention(d_model,n_heads,n_kv_heads,dropout)
        self.n2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model)
    def forward(self,x,cos,sin,mask):
        x = x + self.attn(self.n1(x),cos,sin,mask)
        x = x + self.mlp(self.n2(x))
        return x