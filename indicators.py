"""
Implements common technical indicators for trading strategies
"""
import pandas as pd
import numpy as np


def sma(series: pd.Series, period: int) -> pd.Series:
    """
    Simple Moving Average
    
    Args:
        series: Price series (typically close prices)
        period: Number of periods for the moving average
    
    Returns:
        pd.Series: Simple moving average values
    """
    return series.rolling(window=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """
    Exponential Moving Average
    
    Args:
        series: Price series (typically close prices)
        period: Number of periods for the moving average
    
    Returns:
        pd.Series: Exponential moving average values
    """
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index
    
    Args:
        series: Price series (typically close prices)
        period: Number of periods for RSI calculation (default: 14)
    
    Returns:
        pd.Series: RSI values (0-100)
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi_values = 100 - (100 / (1 + rs))
    
    return rsi_values


def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    """
    Bollinger Bands
    
    Args:
        series: Price series (typically close prices)
        period: Number of periods for moving average (default: 20)
        std_dev: Number of standard deviations (default: 2.0)
    
    Returns:
        tuple: (upper_band, middle_band, lower_band)
    """
    middle_band = sma(series, period)
    std = series.rolling(window=period).std()
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    
    return upper_band, middle_band, lower_band


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """
    Moving Average Convergence Divergence
    
    Args:
        series: Price series (typically close prices)
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line period (default: 9)
    
    Returns:
        tuple: (macd_line, signal_line, histogram)
    """
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Average True Range
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: Number of periods (default: 14)
    
    Returns:
        pd.Series: ATR values
    """
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_values = tr.rolling(window=period).mean()
    
    return atr_values


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
               period: int = 14, smooth_k: int = 3, smooth_d: int = 3):
    """
    Stochastic Oscillator
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: Number of periods (default: 14)
        smooth_k: Smoothing for %K (default: 3)
        smooth_d: Smoothing for %D (default: 3)
    
    Returns:
        tuple: (%K, %D)
    """
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    
    k_fast = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    k = k_fast.rolling(window=smooth_k).mean()
    d = k.rolling(window=smooth_d).mean()
    
    return k, d


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Volume Weighted Average Price
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        volume: Volume series
    
    Returns:
        pd.Series: VWAP values
    """
    typical_price = (high + low + close) / 3
    vwap_values = (typical_price * volume).cumsum() / volume.cumsum()
    
    return vwap_values


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On Balance Volume
    
    Args:
        close: Close price series
        volume: Volume series
    
    Returns:
        pd.Series: OBV values
    """
    obv_values = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv_values
