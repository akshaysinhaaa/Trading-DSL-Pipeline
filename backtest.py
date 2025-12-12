"""
Backtest Simulator Module
Executes trading strategy signals and calculates performance metrics
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict


@dataclass
class Trade:
    """Represents a single trade"""
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    profit_loss: float
    return_pct: float
    
    def to_dict(self):
        return asdict(self)


class BacktestSimulator:
    """
    Simulates trading strategy execution and calculates performance metrics
    """

    def __init__(self, initial_capital: float = 10000.0):
        """
        Initialize backtest simulator
        
        Args:
            initial_capital: Starting capital for simulation
        """
        self.initial_capital = initial_capital
        self.trades: List[Trade] = []
    
    def run(self, df: pd.DataFrame, signals: pd.DataFrame) -> Dict[str, any]:
        """
        Run backtest simulation
        
        Args:
            df: DataFrame with OHLCV data
            signals: DataFrame with 'entry' and 'exit' boolean columns
        
        Returns:
            dict: Backtest results and performance metrics
        """
        self.trades = []
        
        # State variables
        in_position = False
        entry_idx = None
        entry_price = None
        
        # Track equity curve
        equity = [self.initial_capital]
        current_capital = self.initial_capital
        
        # Iterate through signals
        for i in range(len(df)):
            if not in_position:
                # Check for entry signal
                if signals.iloc[i]['entry']:
                    in_position = True
                    entry_idx = i
                    entry_price = df.iloc[i]['close']
            else:
                # Check for exit signal
                if signals.iloc[i]['exit']:
                    in_position = False
                    exit_price = df.iloc[i]['close']
                    
                    # Calculate trade results
                    profit_loss = exit_price - entry_price
                    return_pct = (profit_loss / entry_price) * 100
                    
                    # Update capital
                    current_capital += profit_loss * (current_capital / entry_price)
                    
                    # Record trade
                    trade = Trade(
                        entry_date=str(df.index[entry_idx]),
                        exit_date=str(df.index[i]),
                        entry_price=entry_price,
                        exit_price=exit_price,
                        profit_loss=profit_loss,
                        return_pct=return_pct
                    )
                    self.trades.append(trade)
            
            # Update equity curve
            if in_position:
                current_price = df.iloc[i]['close']
                current_equity = current_capital + (current_price - entry_price) * (current_capital / entry_price)
                equity.append(current_equity)
            else:
                equity.append(current_capital)
        
        # Calculate performance metrics
        metrics = self._calculate_metrics(equity)
        
        # Prepare results
        results = {
            'metrics': metrics,
            'trades': [trade.to_dict() for trade in self.trades],
            'equity_curve': equity
        }
        
        return results
    
    def _calculate_metrics(self, equity: List[float]) -> Dict[str, float]:
        """
        Calculate performance metrics
        
        Args:
            equity: List of equity values over time
        
        Returns:
            dict: Performance metrics
        """
        equity_series = pd.Series(equity)
        
        # Total return
        total_return = ((equity[-1] - self.initial_capital) / self.initial_capital) * 100
        
        # Max drawdown
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()
        
        # Win rate
        if self.trades:
            winning_trades = sum(1 for t in self.trades if t.profit_loss > 0)
            win_rate = (winning_trades / len(self.trades)) * 100
            
            # Average win/loss
            wins = [t.profit_loss for t in self.trades if t.profit_loss > 0]
            losses = [t.profit_loss for t in self.trades if t.profit_loss <= 0]
            
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            
            # Profit factor
            total_wins = sum(wins) if wins else 0
            total_losses = abs(sum(losses)) if losses else 0
            profit_factor = total_wins / total_losses if total_losses > 0 else 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        # Sharpe ratio (simplified, assuming daily data)
        returns = equity_series.pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0
        
        metrics = {
            'total_return': round(total_return, 2),
            'max_drawdown': round(max_drawdown, 2),
            'num_trades': len(self.trades),
            'win_rate': round(win_rate, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'profit_factor': round(profit_factor, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'final_capital': round(equity[-1], 2)
        }
        
        return metrics
    
    def print_results(self, results: Dict[str, any]):
        """
        Print backtest results in a formatted way
        
        Args:
            results: Results dictionary from run()
        """
        metrics = results['metrics']
        trades = results['trades']
        
        print("=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"\nPerformance Metrics:")
        print(f"  Initial Capital:    ${self.initial_capital:,.2f}")
        print(f"  Final Capital:      ${metrics['final_capital']:,.2f}")
        print(f"  Total Return:       {metrics['total_return']:+.2f}%")
        print(f"  Max Drawdown:       {metrics['max_drawdown']:.2f}%")
        print(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:.2f}")
        print(f"\nTrading Statistics:")
        print(f"  Number of Trades:   {metrics['num_trades']}")
        print(f"  Win Rate:           {metrics['win_rate']:.2f}%")
        print(f"  Average Win:        ${metrics['avg_win']:+.2f}")
        print(f"  Average Loss:       ${metrics['avg_loss']:+.2f}")
        print(f"  Profit Factor:      {metrics['profit_factor']:.2f}")
        
        if trades:
            print(f"\n{'=' * 60}")
            print("Trade Log:")
            print(f"{'=' * 60}")
            print(f"{'Entry Date':<20} {'Exit Date':<20} {'Entry $':<10} {'Exit $':<10} {'P/L $':<10} {'Return %':<10}")
            print("-" * 60)
            
            for trade in trades:
                print(f"{trade['entry_date']:<20} {trade['exit_date']:<20} "
                      f"{trade['entry_price']:<10.2f} {trade['exit_price']:<10.2f} "
                      f"{trade['profit_loss']:<+10.2f} {trade['return_pct']:<+10.2f}")
        
        print("=" * 60)


if __name__ == "__main__":
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'open': 100 + np.random.randn(100).cumsum(),
        'high': 100 + np.random.randn(100).cumsum() + 2,
        'low': 100 + np.random.randn(100).cumsum() - 2,
        'close': 100 + np.random.randn(100).cumsum(),
        'volume': np.random.randint(900000, 1500000, 100)
    }, index=dates)

    signals = pd.DataFrame({
        'entry': [False] * 10 + [True] + [False] * 20 + [True] + [False] * 68,
        'exit': [False] * 20 + [True] + [False] * 20 + [True] + [False] * 58
    }, index=dates)

    simulator = BacktestSimulator(initial_capital=10000)
    results = simulator.run(df, signals)
    simulator.print_results(results)
