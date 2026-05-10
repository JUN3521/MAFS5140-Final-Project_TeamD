import pandas as pd
import numpy as np
from collections import deque
import warnings

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ===================== Adaptive Multi-Factor Strategy =====================

class AdaptiveMultiFactorStrategy:
    """
    Adaptive Multi-Factor Strategy with Market Regime Detection

    Factors:
      1. Short-term Reversal     (buy recent losers)
      2. Medium-term Momentum    (buy recent winners)
      3. Volume Confirmation     (abnormal volume)
    """

    def __init__(self, tickers, top_k=20):
        self.tickers = list(tickers)
        self.n_stocks = len(tickers)
        self.top_k = top_k
        self.spy_ticker = 'SPY'
        self.spy_idx = self.tickers.index(self.spy_ticker) if self.spy_ticker in self.tickers else 0

        self.lookback_short = 6        # 30 min reversal
        self.lookback_medium = 14      # 70 min momentum
        self.vol_lookback = 20         # for volatility filter

        max_lookback = max(self.lookback_short, self.lookback_medium, self.vol_lookback) + 5
        self.price_history = deque(maxlen=max_lookback)
        self.volume_history = deque(maxlen=max_lookback)
        self.step_count = 0
        self.last_weights = np.zeros(self.n_stocks)

        # EMA smoothing for composite scores (reduces noise & turnover)
        self.composite_ema = None
        self.ema_alpha = 0.05

    def step(self, prices_arr, volumes_arr):
        self.price_history.append(prices_arr.copy())
        self.volume_history.append(volumes_arr.copy())
        self.step_count += 1

        min_history = self.lookback_medium + 1
        if self.step_count < min_history:
            return self.last_weights

        price_arr = np.array(self.price_history)
        vol_arr = np.array(self.volume_history)
        current_prices = price_arr[-1]

        # Factor 1: Short-term reversal (buy losers)
        past_short = price_arr[-1 - self.lookback_short]
        ret_short = (current_prices - past_short) / (past_short + 1e-10)
        reversal_signal = -ret_short

        # Factor 2: Medium-term momentum (buy winners)
        past_medium = price_arr[-1 - self.lookback_medium]
        ret_medium = (current_prices - past_medium) / (past_medium + 1e-10)
        momentum_signal = ret_medium

        # Factor 3: Volume confirmation (z-score of relative volume)
        avg_vol = np.nanmean(vol_arr[-10:], axis=0)
        std_vol = np.nanstd(vol_arr[-10:], axis=0)
        volume_signal = (volumes_arr - avg_vol) / (std_vol + 1e-10)
        volume_signal = np.clip(volume_signal, -3, 3)

        # --- Market Regime Detection (SPY proxy) ---
        spy_5bar = (price_arr[-1, self.spy_idx] - price_arr[-6, self.spy_idx]) / price_arr[-6, self.spy_idx] if self.step_count >= 6 else 0
        spy_medium = ret_medium[self.spy_idx]

        # Adaptive factor weights (momentum-heavy for bull markets)
        if spy_5bar > 0.001 or spy_medium > 0.003:
            w_rev, w_mom, w_vol = 0.05, 0.90, 0.05
        elif spy_5bar < -0.001 or spy_medium < -0.003:
            w_rev, w_mom, w_vol = 0.80, 0.15, 0.05
        else:
            w_rev, w_mom, w_vol = 0.15, 0.80, 0.05

        # Rank-based composite scoring
        rank_rev = self._rank(reversal_signal)
        rank_mom = self._rank(momentum_signal)
        rank_vol = self._rank(volume_signal)
        composite = (w_rev * rank_rev +
                     w_mom * rank_mom +
                     w_vol * rank_vol)

        # EMA smoothing of composite score
        if self.composite_ema is None:
            self.composite_ema = composite.copy()
        else:
            self.composite_ema = self.ema_alpha * composite + (1 - self.ema_alpha) * self.composite_ema
        composite = self.composite_ema

        # Volatility Filter: exclude highest-volatility stocks in bear markets
        is_bear = (spy_5bar < -0.001 or spy_medium < -0.003)
        if is_bear and self.step_count > self.vol_lookback:
            recent = price_arr[-self.vol_lookback:]
            rets = np.diff(recent, axis=0) / (recent[:-1] + 1e-10)
            stock_vols = np.nanstd(rets, axis=0)
            vol_thresh = np.nanquantile(stock_vols, 0.85)
            composite = np.where(stock_vols > vol_thresh, -np.inf, composite)

        # Build target weights with holding priority
        sorted_idx = np.argsort(composite)
        target = np.zeros(self.n_stocks)

        if np.sum(self.last_weights) > 0:
            current_holdings = np.where(self.last_weights > 0.001)[0]
            keep_mask = np.zeros(self.n_stocks, dtype=bool)
            for stock in current_holdings:
                rank_pos = np.where(sorted_idx == stock)[0]
                if len(rank_pos) > 0 and rank_pos[0] >= self.n_stocks - 50:
                    keep_mask[stock] = True
            n_new = self.top_k - keep_mask.sum()
            if n_new > 0:
                new_stocks = sorted_idx[-n_new:]
                keep_mask[new_stocks] = True
            keep_stocks = np.where(keep_mask)[0]
            if len(keep_stocks) > 0:
                target[keep_stocks] = 1.0 / len(keep_stocks)
        else:
            top_indices = sorted_idx[-self.top_k:]
            target[top_indices] = 1.0 / self.top_k

        # Smooth weight transition: cap single-rebalance turnover at 30%
        turnover = np.abs(target - self.last_weights).sum()
        max_turnover = 0.30
        if turnover > max_turnover and np.sum(self.last_weights) > 0:
            blend = max_turnover / turnover
            weights_arr = blend * target + (1 - blend) * self.last_weights
        else:
            weights_arr = target.copy()

        return weights_arr

    @staticmethod
    def _rank(arr):
        clean = np.where(np.isfinite(arr), arr, -np.inf)
        temp = clean.argsort().argsort()
        return temp / (len(temp) - 1 + 1e-10)


# ===================== Core Metrics Evaluator =====================

class Evaluator:
    def __init__(self, returns: pd.Series, periods_per_year: int = 252):
        """
        periods_per_year: 252 for daily data, 252*78 for 5-min data, etc.
        """
        self.returns = returns.dropna()
        self.periods_per_year = periods_per_year

    def cumulative_return(self) -> float:
        return (1 + self.returns).prod() - 1.0

    def cagr(self) -> float:
        """Geometric Annualized Return (CAGR)."""
        cum_ret = self.cumulative_return()
        num_periods = self.returns.count()
        if num_periods == 0:
            return 0.0
        return (1 + cum_ret) ** (self.periods_per_year / num_periods) - 1.0

    def annualized_volatility(self) -> float:
        return self.returns.std() * np.sqrt(self.periods_per_year)

    def sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Annualized Sharpe Ratio (numerator uses Arithmetic Mean)."""
        ann_vol = self.annualized_volatility()
        if ann_vol == 0:
            return 0.0
        arithmetic_ann_ret = self.returns.mean() * self.periods_per_year
        return (arithmetic_ann_ret - risk_free_rate) / ann_vol

    def max_drawdown(self) -> float:
        """Maximum Drawdown."""
        cumulative_wealth = (1 + self.returns).cumprod()
        wealth_with_initial = pd.concat([pd.Series([1.0]), cumulative_wealth], ignore_index=True)
        rolling_max = wealth_with_initial.cummax()
        drawdowns = (wealth_with_initial - rolling_max) / rolling_max
        return drawdowns.min()

    def generate_report(self):
        """Prints and returns core metrics dictionary."""
        metrics = {
            "Cumulative Return": f"{self.cumulative_return() * 100:.4f}%",
            "CAGR (Annualized Return)": f"{self.cagr() * 100:.4f}%",
            "Annualized Volatility": f"{self.annualized_volatility() * 100:.4f}%",
            "Sharpe Ratio": f"{self.sharpe_ratio():.4f}",
            "Max Drawdown": f"{self.max_drawdown() * 100:.4f}%",
        }

        print("=" * 50)
        print("--- Strategy Core Metrics Report ---")
        for key, value in metrics.items():
            print(f"{key:<30}: {value}")
        print("-" * 50)
        return metrics


# ===================== Backtest Runner =====================

def run_backtest(data_path: str, top_k: int = 20):
    """Run strategy backtest and return portfolio / benchmark returns."""
    # Load data
    df = pd.read_parquet(data_path)
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

    P = price_data.values
    V = volume_data.values
    idx = price_data.index

    # Setup strategy
    strategy = AdaptiveMultiFactorStrategy(tickers=price_data.columns.tolist(), top_k=top_k)
    REBALANCE_TIMES = [(11, 0), (13, 0), (15, 55)]
    weights = np.zeros_like(P)
    current_weights = np.zeros(P.shape[1])

    # Backtest loop
    for i in range(len(P)):
        ts = idx[i]
        w = strategy.step(P[i], V[i])

        if (ts.hour, ts.minute) in REBALANCE_TIMES:
            # SPY intraday filter: if SPY is down > 0.5% from open, halve position
            day_start_idx = None
            for j in range(i, -1, -1):
                if idx[j].hour == 9 and idx[j].minute == 30:
                    day_start_idx = j
                    break
            if day_start_idx is not None:
                spy_intraday = (P[i, strategy.spy_idx] - P[day_start_idx, strategy.spy_idx]) / P[day_start_idx, strategy.spy_idx]
                if spy_intraday < -0.005:
                    w = w * 0.5
            current_weights = w.copy()
            strategy.last_weights = current_weights.copy()

        weights[i] = current_weights.copy()

    # Portfolio & benchmark returns (NO transaction cost)
    ret_matrix = np.diff(P, axis=0) / (P[:-1] + 1e-10)
    portfolio_returns = pd.Series(np.nansum(weights[:-1] * ret_matrix, axis=1), index=idx[1:]).fillna(0)
    spy_returns = pd.Series(ret_matrix[:, strategy.spy_idx], index=idx[1:])

    # Hit ratio (5-min level)
    win_rate = (portfolio_returns > 0).mean()

    # Turnover (at rebalance points only)
    rebalance_mask = pd.Series(idx).apply(lambda x: (x.hour, x.minute) in REBALANCE_TIMES).values
    rebalance_idx = np.where(rebalance_mask)[0]
    turnover_values = []
    for i in rebalance_idx[1:]:
        turnover_values.append(np.sum(np.abs(weights[i] - weights[i - 1])))
    turnover = np.mean(turnover_values) if len(turnover_values) > 0 else 0.0
    annual_turnover = turnover * 252 * len(REBALANCE_TIMES)  # 3 rebalances per day

    return portfolio_returns, spy_returns, {
        "price_data": price_data,
        "win_rate": win_rate,
        "annual_turnover": annual_turnover,
    }


# ===================== Main =====================

if __name__ == "__main__":
    DATA_PATH = '/Users/taojun/Desktop/5140 Final_TeamD/validation.parquet'
    PERIODS_PER_YEAR = 252 * 78  # 5-min bars

    print("Running backtest...")
    portfolio_returns, spy_returns, info = run_backtest(DATA_PATH, top_k=20)

    print(f"\nData Source             : {DATA_PATH.split('/')[-1]}")
    print(f"Period Start            : {info['price_data'].index[0]}")
    print(f"Period End              : {info['price_data'].index[-1]}")
    print(f"Number of Time Points   : {len(info['price_data'].index)}")
    print(f"Number of Stocks Held   : 20")

    # Core metrics via Evaluator
    evaluator = Evaluator(portfolio_returns, periods_per_year=PERIODS_PER_YEAR)
    evaluator.generate_report()

    # Supplementary stats
    print(f"{'Hit Ratio (Win Rate)':<30}: {info['win_rate'] * 100:.2f}%")
    print(f"{'Est. Annual Turnover':<30}: {info['annual_turnover']:.2f}x")

    # Benchmark (SPY)
    spy_eval = Evaluator(spy_returns, periods_per_year=PERIODS_PER_YEAR)
    print(f"{'Benchmark (SPY) Cum. Return':<30}: {spy_eval.cumulative_return() * 100:.2f}%")
    print(f"{'Benchmark (SPY) Ann. Return':<30}: {spy_eval.cagr() * 100:.2f}%")
    print(f"{'Benchmark (SPY) Sharpe':<30}: {spy_eval.sharpe_ratio():.4f}")
    print("=" * 50)
