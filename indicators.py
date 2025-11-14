# indicators.py
import pandas as pd
import ta

def money_flow_indicators(df):
    mfi = ta.volume.money_flow_index(
        high=df["high"], low=df["low"],
        close=df["close"], volume=df["volume"], window=14
    )
    obv = ta.volume.on_balance_volume(df["close"], df["volume"])
    return mfi, obv

def macd_signals_full(df: pd.DataFrame):
    macd = ta.trend.macd(df["close"])
    signal = ta.trend.macd_signal(df["close"])
    cross_up = (macd.shift(1) < signal.shift(1)) & (macd > signal)
    cross_down = (macd.shift(1) > signal.shift(1)) & (macd < signal)

    buys = df.loc[cross_up, ["time", "close"]]
    sells = df.loc[cross_down, ["time", "close"]]
    return buys, sells, macd, signal


def rsi_signals(df: pd.DataFrame, low_th=30, high_th=70):
    rsi = ta.momentum.rsi(df["close"], window=14)

    cond_buy = (rsi.shift(1) < low_th) & (rsi >= low_th)
    cond_sell = (rsi.shift(1) > high_th) & (rsi <= high_th)

    buys = df.loc[cond_buy, ["time", "close"]]
    sells = df.loc[cond_sell, ["time", "close"]]
    return buys, sells, rsi


def stochastic_signals(df: pd.DataFrame):
    stoch_k = ta.momentum.stoch(df["high"], df["low"], df["close"])
    stoch_d = ta.momentum.stoch_signal(df["high"], df["low"], df["close"])

    cond_buy = (stoch_k.shift(1) < stoch_d.shift(1)) & (stoch_k > stoch_d) & (stoch_k < 20)
    cond_sell = (stoch_k.shift(1) > stoch_d.shift(1)) & (stoch_k < stoch_d) & (stoch_k > 80)

    buys = df.loc[cond_buy, ["time", "close"]]
    sells = df.loc[cond_sell, ["time", "close"]]
    return buys, sells, stoch_k, stoch_d


def adx_di_signals(df: pd.DataFrame):
    adx = ta.trend.adx(df["high"], df["low"], df["close"])
    di_pos = ta.trend.adx_pos(df["high"], df["low"], df["close"])
    di_neg = ta.trend.adx_neg(df["high"], df["low"], df["close"])

    cond_buy = (di_pos.shift(1) < di_neg.shift(1)) & (di_pos > di_neg) & (adx > 25)
    cond_sell = (di_pos.shift(1) > di_neg.shift(1)) & (di_pos < di_neg) & (adx > 25)

    buys = df.loc[cond_buy, ["time", "close"]]
    sells = df.loc[cond_sell, ["time", "close"]]
    return buys, sells, adx, di_pos, di_neg


def golden_cross_signals(df: pd.DataFrame, short_win=50, long_win=200):
    sma_s = ta.trend.sma_indicator(df["close"], window=short_win)
    sma_l = ta.trend.sma_indicator(df["close"], window=long_win)

    cross_up   = (sma_s.shift(1) < sma_l.shift(1)) & (sma_s > sma_l)
    cross_down = (sma_s.shift(1) > sma_l.shift(1)) & (sma_s < sma_l)

    buys  = df.loc[cross_up,  ["time", "close"]]
    sells = df.loc[cross_down,["time", "close"]]
    return buys, sells, sma_s, sma_l
