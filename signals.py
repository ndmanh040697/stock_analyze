# signals.py
import pandas as pd
import numpy as np


def evaluate_signals(buys: pd.DataFrame, sells: pd.DataFrame):
    """Backtest tín hiệu giao dịch: mỗi lệnh Buy ghép với Sell tiếp theo."""
    if buys is None or sells is None or len(buys) == 0 or len(sells) == 0:
        return pd.DataFrame(), {
            "Strategy": "",
            "TotalTrades": 0,
            "WinRate": np.nan,
            "AvgGain": np.nan,
            "AvgLoss": np.nan,
            "Expectancy": np.nan,
        }

    buys_sorted = buys.sort_values("time").reset_index(drop=True)
    sells_sorted = sells.sort_values("time").reset_index(drop=True)

    trades = []
    j = 0

    for i in range(len(buys_sorted)):
        buy_row = buys_sorted.iloc[i]
        buy_time = buy_row["time"]

        while j < len(sells_sorted) and sells_sorted.iloc[j]["time"] <= buy_time:
            j += 1

        if j >= len(sells_sorted):
            break

        sell_row = sells_sorted.iloc[j]
        sell_time = sell_row["time"]

        buy_price = buy_row["close"]
        sell_price = sell_row["close"]
        ret = (sell_price - buy_price) / buy_price * 100

        trades.append(
            {
                "Buy": buy_time,
                "Sell": sell_time,
                "BuyPrice": buy_price,
                "SellPrice": sell_price,
                "Return(%)": ret,
            }
        )
        j += 1

    if not trades:
        return pd.DataFrame(), {
            "Strategy": "",
            "TotalTrades": 0,
            "WinRate": np.nan,
            "AvgGain": np.nan,
            "AvgLoss": np.nan,
            "Expectancy": np.nan,
        }

    df_trades = pd.DataFrame(trades)
    wins = df_trades[df_trades["Return(%)"] > 0]
    losses = df_trades[df_trades["Return(%)"] <= 0]

    win_rate = len(wins) / len(df_trades) * 100
    avg_gain = wins["Return(%)"].mean() if len(wins) else 0
    avg_loss = abs(losses["Return(%)"].mean()) if len(losses) else 0
    expectancy = (win_rate / 100) * avg_gain - (1 - win_rate / 100) * avg_loss

    summary = {
        "Strategy": "",
        "TotalTrades": len(df_trades),
        "WinRate": win_rate,
        "AvgGain": avg_gain,
        "AvgLoss": avg_loss,
        "Expectancy": expectancy,
    }
    return df_trades, summary
