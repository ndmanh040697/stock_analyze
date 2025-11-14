# utils.py
import numpy as np
import plotly.graph_objects as go
import pandas as pd


def plot_price_with_bands(df, title, buys=None, sells=None, extra_lines=None,
                          fc_idx=None, yhat=None, low=None, up=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["time"], y=df["close"],
                             name="Close", mode="lines+markers"))

    if extra_lines:
        for line in extra_lines:
            fig.add_trace(go.Scatter(
                x=df["time"], y=line["y"],
                name=line["name"], mode="lines"
            ))

    if buys is not None and len(buys):
        fig.add_trace(go.Scatter(
            x=buys["time"], y=buys["close"],
            mode="markers", name="Buy",
            marker=dict(symbol="triangle-up", size=12, color="green")
        ))
    if sells is not None and len(sells):
        fig.add_trace(go.Scatter(
            x=sells["time"], y=sells["close"],
            mode="markers", name="Sell",
            marker=dict(symbol="triangle-down", size=12, color="red")
        ))

    if fc_idx is not None and yhat is not None:
        fig.add_trace(go.Scatter(
            x=fc_idx, y=yhat,
            name="Forecast", mode="lines",
            line=dict(dash="dash", color="magenta")
        ))
        if low is not None and up is not None:
            fig.add_trace(go.Scatter(
                x=list(fc_idx) + list(fc_idx[::-1]),
                y=list(up) + list(low[::-1]),
                fill="toself", name="95% CI",
                line=dict(color="violet"), opacity=0.2
            ))

            x_last = fc_idx[-1]
            fig.add_annotation(x=x_last, y=yhat.iloc[-1],
                               text=f"{yhat.iloc[-1]:.2f}",
                               showarrow=False,
                               font=dict(color="magenta", size=12))
            fig.add_annotation(x=x_last, y=up.iloc[-1],
                               text=f"{up.iloc[-1]:.2f}",
                               showarrow=False,
                               font=dict(color="green", size=11))
            fig.add_annotation(x=x_last, y=low.iloc[-1],
                               text=f"{low.iloc[-1]:.2f}",
                               showarrow=False,
                               font=dict(color="red", size=11))

    fig.update_layout(title=title, legend=dict(orientation="h"), height=520)
    return fig


def confidence_color(mape):
    if np.isnan(mape):
        return "gray"
    if mape <= 10:
        return "green"
    if mape <= 20:
        return "gold"
    return "red"
