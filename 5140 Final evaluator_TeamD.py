import pandas as pd
import numpy as np
from collections import deque
import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ===================== Evaluator (from evaluator.py) =====================

class Evaluator:
    def __init__(self, returns: pd.Series, periods_per_year: int = 252):
        self.returns = returns.dropna()
        self.periods_per_year = periods_per_year

    def cumulative_return(self) -> float:
        return (1 + self.returns).prod() - 1.0

    def cagr(self) -> float:
        cum_ret = self.cumulative_return()
        num_periods = self.returns.count()
        if num_periods == 0:
            return 0.0
        return (1 + cum_ret) ** (self.periods_per_year / num_periods) - 1.0

    def annualized_volatility(self) -> float:
        return self.returns.std() * np.sqrt(self.periods_per_year)

    def sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        ann_vol = self.annualized_volatility()
        if ann_vol == 0:
            return 0.0
        arithmetic_ann_ret = self.returns.mean() * self.periods_per_year
        return (arithmetic_ann_ret - risk_free_rate) / ann_vol

    def max_drawdown(self) -> float:
        cumulative_wealth = (1 + self.returns).cumprod()
        wealth_with_initial = pd.concat([pd.Series([1.0]), cumulative_wealth], ignore_index=True)
        rolling_max = wealth_with_initial.cummax()
        drawdowns = (wealth_with_initial - rolling_max) / rolling_max
        return drawdowns.min()

    def generate_report(self, label: str = "Strategy"):
        metrics = {
            "Cumulative Return": f"{self.cumulative_return() * 100:.4f}%",
            "CAGR": f"{self.cagr() * 100:.4f}%",
            "Annualized Volatility": f"{self.annualized_volatility() * 100:.4f}%",
            "Sharpe Ratio": f"{self.sharpe_ratio():.4f}",
            "Max Drawdown": f"{self.max_drawdown() * 100:.4f}%"
        }

        print(f"\n--- {label} Performance Report ---")
        for key, value in metrics.items():
            print(f"{key:<25}: {value}")
        print("-" * 35)
        return metrics


# ===================== Strategy Design =====================

class AdaptiveMultiFactorStrategy:
    def __init__(self, tickers, top_k=20):
        self.tickers = list(tickers)
        self.n_stocks = len(tickers)
        self.top_k = top_k
        self.spy_ticker = 'SPY'
        self.spy_idx = self.tickers.index(self.spy_ticker) if self.spy_ticker in self.tickers else 0

        self.lookback_short = 6
        self.lookback_medium = 14
        self.vol_lookback = 20

        max_lookback = max(self.lookback_short, self.lookback_medium, self.vol_lookback) + 5
        self.price_history = deque(maxlen=max_lookback)
        self.volume_history = deque(maxlen=max_lookback)
        self.step_count = 0

    def step(self, prices_arr, volumes_arr):
        self.price_history.append(prices_arr.copy())
        self.volume_history.append(volumes_arr.copy())
        self.step_count += 1

        min_history = self.lookback_medium + 1
        if self.step_count < min_history:
            return np.zeros(self.n_stocks)

        price_arr = np.array(self.price_history)
        vol_arr = np.array(self.volume_history)
        current_prices = price_arr[-1]

        past_short = price_arr[-1 - self.lookback_short]
        ret_short = (current_prices - past_short) / (past_short + 1e-10)
        reversal_signal = -ret_short

        past_medium = price_arr[-1 - self.lookback_medium]
        ret_medium = (current_prices - past_medium) / (past_medium + 1e-10)
        momentum_signal = ret_medium

        avg_vol = np.nanmean(vol_arr[-10:], axis=0)
        std_vol = np.nanstd(vol_arr[-10:], axis=0)
        volume_signal = (volumes_arr - avg_vol) / (std_vol + 1e-10)
        volume_signal = np.clip(volume_signal, -3, 3)

        spy_5bar = (price_arr[-1, self.spy_idx] - price_arr[-6, self.spy_idx]) / price_arr[-6, self.spy_idx] if self.step_count >= 6 else 0
        spy_medium = ret_medium[self.spy_idx]

        if spy_5bar > 0.001 or spy_medium > 0.003:
            w_rev, w_mom, w_vol = 0.50, 0.45, 0.05
        elif spy_5bar < -0.001 or spy_medium < -0.003:
            w_rev, w_mom, w_vol = 0.80, 0.15, 0.05
        else:
            w_rev, w_mom, w_vol = 0.90, 0.05, 0.05

        rank_rev = self._rank(reversal_signal)
        rank_mom = self._rank(momentum_signal)
        rank_vol = self._rank(volume_signal)
        composite = (w_rev * rank_rev + w_mom * rank_mom + w_vol * rank_vol)

        is_bear = (spy_5bar < -0.001 or spy_medium < -0.003)
        if is_bear and self.step_count > self.vol_lookback:
            recent = price_arr[-self.vol_lookback:]
            rets = np.diff(recent, axis=0) / (recent[:-1] + 1e-10)
            stock_vols = np.nanstd(rets, axis=0)
            vol_thresh = np.nanquantile(stock_vols, 0.85)
            composite = np.where(stock_vols > vol_thresh, -np.inf, composite)

        top_indices = np.argsort(composite)[-self.top_k:]
        weights_arr = np.zeros(self.n_stocks)
        weights_arr[top_indices] = 1.0 / self.top_k

        return weights_arr

    @staticmethod
    def _rank(arr):
        clean = np.where(np.isfinite(arr), arr, -np.inf)
        temp = clean.argsort().argsort()
        return temp / (len(temp) - 1 + 1e-10)


# ===================== Data Loading =====================

DATA_PATH = '/Users/taojun/Desktop/train.parquet'

df = pd.read_parquet(DATA_PATH)
df.index = pd.to_datetime(df.index)

price_data = pd.DataFrame({col[0]: df[col] for col in df.columns
                           if isinstance(col, tuple) and col[1] == 'close'})
volume_data = pd.DataFrame({col[0]: df[col] for col in df.columns
                            if isinstance(col, tuple) and col[1] == 'volume'})

common_stocks = list(set(price_data.columns) & set(volume_data.columns))
price_data = price_data[common_stocks]
volume_data = volume_data[common_stocks]

price_data = price_data.ffill(axis=0)
volume_data = volume_data.fillna(0)

print(f"Loaded: {DATA_PATH}")
print(f"Shape: {df.shape}")
print(f"Stocks: {len(common_stocks)}")
print(f"Date range: {price_data.index[0]} to {price_data.index[-1]}")


# ===================== Backtesting =====================

P = price_data.values
V = volume_data.values
idx = price_data.index

strategy = AdaptiveMultiFactorStrategy(tickers=price_data.columns.tolist(), top_k=20)

weights = np.zeros_like(P)
for i in range(len(P)):
    weights[i] = strategy.step(P[i], V[i])

ret_matrix = np.diff(P, axis=0) / (P[:-1] + 1e-10)
portfolio_returns = pd.Series(np.nansum(weights[:-1] * ret_matrix, axis=1), index=idx[1:]).fillna(0)


# ===================== Evaluation =====================

PERIODS_PER_YEAR = 252 * 78  # 5-min bars

strategy_eval = Evaluator(portfolio_returns, periods_per_year=PERIODS_PER_YEAR)
_ = strategy_eval.generate_report(label="Strategy")
