import pandas as pd

"""Relative Strength Index"""

def rsi (df, periods = 14):
    """
    Returns a pd.Series with the relative strength index.
    """
    close_delta = df['Close'].diff()

    # Make two series: one for lower closes and one for higher closes
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    # Use simple moving average
    ma_up = up.rolling(window = periods).mean()
    ma_down = down.rolling(window = periods).mean()
        
    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    return rsi

"""Wiliams %R"""

def williams (df, periods = 14):
    """
    Returns a pd.Series with the Williams %R.
    """
    close = df['Close']
    high = df['High']
    low = df['Low']

    # take higher and lower values from rolling window
    higher = high.rolling(window = periods).max()
    lower = low.rolling(window = periods).min()    
        
    wm_r = 100 * ( (higher - close) / (higher - lower) )
    return wm_r

"""Stochastic Oscillator"""

def stochastic (df, periods_K = 14, periods_D = 3):
    """
    Returns two pd.Series, one with the Stochastic %K and other with Stochastic %D.
    """
    close = df['Close']
    high = df['High']
    low = df['Low']

    # take higher and lower values from rolling window
    higher = high.rolling(window = periods_K).max()
    lower = low.rolling(window = periods_K).min()

    K_numerator = (close - lower)
    K_denominator = (higher - lower)

    D_numerator = K_numerator.rolling(window = periods_D).sum()
    D_denominator = K_denominator.rolling(window = periods_D).sum()

    sto_K = 100 * ( K_numerator / K_denominator )
    sto_D = 100 * ( D_numerator / D_denominator )
    
    return sto_K, sto_D