# patterns.py
import pandas as pd

def detect_candlestick_patterns(df: pd.DataFrame):
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    range_ = h - l

    # Doji
    doji = body <= (range_ * 0.1)

    # Hammer: râu dưới dài, thân nhỏ ở trên
    lower_shadow = c.where(c>o, o) - l
    upper_shadow = h - c.where(c>o, o)
    hammer = (lower_shadow > 2*body) & (upper_shadow < body)

    # Shooting star: râu trên dài
    shooting = (upper_shadow > 2*body) & (lower_shadow < body)

    # Bullish Engulfing
    prev_o, prev_c = o.shift(1), c.shift(1)
    bullish_engulf = (prev_c < prev_o) & (c > o) & (c >= prev_o) & (o <= prev_c)

    # Bearish Engulfing
    bearish_engulf = (prev_c > prev_o) & (c < o) & (c <= prev_o) & (o >= prev_c)

    patterns = pd.DataFrame({
        "doji": doji,
        "hammer": hammer,
        "shooting_star": shooting,
        "bull_engulf": bullish_engulf,
        "bear_engulf": bearish_engulf,
    }, index=df.index)
    return patterns
