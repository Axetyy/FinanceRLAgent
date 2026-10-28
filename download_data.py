import os
import numpy as np
import pandas as pd
import yfinance as yf
import config as C

def rsi (close:pd.Series,n:int = 14)->pd.Series:
    delta = close.diff()
    up = delta.clip(lower= 0).ewm(alpha=1/n,adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n,adjust=False).mean()
    rs = up / (down + 1e-12)
    return 100 - 100 / (1+rs)

def atr(df:pd.DataFrame,n:int = 14)->pd.Series:
    prev_close = df['Close'].shift(1)
    tr = pd.concat([df['High']-df['Low'],(df['High']-prev_close).abs(),(df['Low']-prev_close).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()

def macd(close:pd.DataFrame):
    ema12 = close.ewm(span=12,adjust=False).mean()
    ema26 = close.ewm(span=26,adjust=False).mean()
    line = ema12 - ema26
    signal = line.ewm(span=9,adjust=False).mean()
    return line,signal


def build_features(df:pd.DataFrame)->pd.DataFrame:
    """Per-ticker features. Everything uses only past data"""
    out = pd.DataFrame(index=df.index)
    out['Close'] = df['Close']
    out['ret_1'] = np.log(df['Close']).diff()
    out['ret_5'] = np.log(df['Close']).diff(5)
    out["hl_range"] = (df['High']-df['Low']) / df['Close']
    out['oc_change'] = (df['Close'] - df["Open"]) / df["Open"]
    out['vol_chg'] = np.log(df['Volume'].replace(0,np.nan)).diff()
    out['rsi'] = (rsi(df['Close'])-50) / 50
    m,s = macd(df['Close'])
    out['macd'] = m / df['Close']
    out['macd_sig'] = s / df['Close']
    out['atr'] = atr(df) / df['Close']
    out['vol_20'] = out["ret_1"].rolling(20).std()
    return out

FEATURE_COLS = ["ret_1", "ret_5", "hl_range", "oc_change", "vol_chg",
                "rsi", "macd", "macd_sig", "atr", "vol_20"]


def main():
    os.makedirs(C.DATA_DIR,exist_ok=True)
    frames = {}
    for t in C.TICKERS:
        print(f"Downloading {t} ...")
        df = yf.download(t,start=C.START,end=C.END,interval=C.INTERVAL,auto_adjust=True,progress=False)
        if df.empty:
            raise RuntimeError(f"No data returned for {t}")
        if isinstance(df.columns,pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.to_parquet(f"{C.DATA_DIR}/raw_{t}.parquet")
        frames[t] = build_features(df)
    panel = pd.concat(frames,axis=1,names=["ticker","feature"]).dropna()
    panel.index = pd.to_datetime(panel.index)
    
    train = panel.loc[:C.TRAIN_END]
    val = panel.loc[C.TRAIN_END:C.VAL_END].iloc[1:]
    test = panel.loc[C.VAL_END:].iloc[1:]
    
    stats = {}
    for t in C.TICKERS:
        for f in FEATURE_COLS:
            mu,st = train[(t,f)].mean(),train[(t,f)].std() + 1e-9
            stats[(t,f)] = (mu,st)
            
    def norm(d:pd.DataFrame) ->pd.DataFrame:
        d = d.copy()
        for (t,f),(mu,st) in stats.items():
            d[(t,f)] = ((d[(t,f)]-mu)/st).clip(-5,5)
        return d
    for name,d in [('train',train),('val',val),('test',test)]:
        norm(d).to_parquet(f"{C.DATA_DIR}/{name}.parquet")
        print(f"{name}: {len(d)} rows, {d.index[0].date()} -> {d.index[-1].date()}")
    pd.Series({f"{t}|{f}": v for (t, f), v in stats.items()}).to_pickle(
        f"{C.DATA_DIR}/norm_stats.pkl")
    print("Done.")


if __name__ == "__main__":
    main()
    