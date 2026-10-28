TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "SPY"]

INTERVAL = "1d"
START = "2015-01-01"
END = "2026-10-01"

TRAIN_END = "2022-12-31"
VAL_END = "2024-12-31"

DATA_DIR = "data"

WINDOW = 60
FEE_BPS = 1.0 # commision per unit traded
SLIPPAGE_BPS = 2.0 # base slippage in basis points Slippage is the difference between a trader's expected execution price and the actual price at which a trade is completed
IMPACT_COEF = 10.0 # extra slippage growing with trade size ()
EXEC_DELAY = 1 # decision at t, t+1

LONG_ONLY = True
LOGIT_SCALE  = 3.0
DEADBAND = 0.02 # ignore rebalances smaller than 2% of equity per asset (commitment)
CHURN_PENALTY = 0.001
DD_PENALTY = 0.1 #  penalty weight on NEW drawdown (only when equity falls further below peak)
REWARD_SCALE = 100.0 # 1% log growth -> reward 1.0 (keeps PPO targets well-scaled)

AGE_CAP = 252  # holding-age feature saturates at 1 trading year
NEWS_AGE_CAP = 20 # news-age feature saturates at 20 steps
AUX_HORIZONS = (1, 5, 20)
EPISODE_LEN =  252

PPO_GAMMA = 0.995