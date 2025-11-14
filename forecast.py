# forecast.py
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

try:
    import tensorflow as tf
    HAS_TF = True
except Exception:
    HAS_TF = False

try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False

def _prepare_dl_series(series: pd.Series, window=20):
    """Chuẩn hóa và tạo (X, y) cho DL; chỉ dùng train nên không leak tương lai."""
    arr = series.values.astype("float32")
    # scale 0–1
    min_v, max_v = arr.min(), arr.max()
    scale = max_v - min_v if max_v != min_v else 1.0
    arr_n = (arr - min_v) / scale

    X, y = [], []
    for i in range(window, len(arr_n)):
        X.append(arr_n[i-window:i])
        y.append(arr_n[i])
    X = np.array(X)[..., np.newaxis]   # (samples, window, 1)
    y = np.array(y)
    return X, y, (min_v, scale)

def lstm_forecast(series: pd.Series,
                  steps: int = 50,
                  window: int = 20,
                  epochs: int = 20,
                  batch_size: int = 16):
    if not HAS_TF:
        raise RuntimeError("Chưa cài tensorflow: pip install tensorflow")

    X, y, (min_v, scale) = _prepare_dl_series(series, window=window)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(window, 1)),
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    # roll-forward forecast như bạn đang làm
    last_seq = ((series.values[-window:].astype("float32") - min_v) / scale).reshape(1, window, 1)
    preds = []
    for _ in range(steps):
        p = model.predict(last_seq, verbose=0)[0, 0]
        preds.append(p)
        last_seq = np.roll(last_seq, -1, axis=1)
        last_seq[0, -1, 0] = p

    preds = np.array(preds) * scale + min_v
    idx = future_index(series.index[-1], steps)
    yhat = pd.Series(preds, index=idx)

    train_pred = model.predict(X, verbose=0).flatten()
    resid = y * scale + min_v - train_pred * scale - min_v
    sd = resid.std()
    low = yhat - 1.96 * sd
    up  = yhat + 1.96 * sd
    return yhat, low, up


def transformer_forecast(series: pd.Series,
                         steps: int = 50,
                         window: int = 20,
                         epochs: int = 20,
                         batch_size: int = 16):
    if not HAS_TF:
        raise RuntimeError("Chưa cài tensorflow")

    X, y, (min_v, scale) = _prepare_dl_series(series, window=window)

    inp = tf.keras.layers.Input(shape=(window, 1))

    pos = tf.range(start=0, limit=window, delta=1)
    pos = tf.keras.layers.Embedding(input_dim=window, output_dim=1)(pos)

    x = inp + pos

    attn = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=8)(x, x)
    x = tf.keras.layers.LayerNormalization()(x + attn)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    out = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    last_seq = ((series.values[-window:].astype("float32") - min_v) / scale).reshape(1, window, 1)
    preds = []
    for _ in range(steps):
        p = model.predict(last_seq, verbose=0)[0, 0]
        preds.append(p)
        last_seq = np.roll(last_seq, -1, axis=1)
        last_seq[0, -1, 0] = p

    preds = np.array(preds) * scale + min_v
    idx = future_index(series.index[-1], steps)
    yhat = pd.Series(preds, index=idx)

    train_pred = model.predict(X, verbose=0).flatten()
    resid = (y * scale + min_v) - (train_pred * scale + min_v)
    sd = resid.std()
    low = yhat - 1.96 * sd
    up  = yhat + 1.96 * sd
    return yhat, low, up



def compute_metrics(actual, pred):
    actual = np.array(actual, dtype=float)
    pred   = np.array(pred, dtype=float)

    n = min(len(actual), len(pred))
    actual = actual[-n:]
    pred   = pred[-n:]

    if n == 0:
        return {"n": 0, "MAPE": np.nan, "MAD": np.nan, "MSD": np.nan}

    mask = actual != 0
    actual = actual[mask]
    pred   = pred[mask]

    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    mad  = np.mean(np.abs(actual - pred))
    msd  = np.mean((actual - pred) ** 2)
    return {"n": len(actual), "MAPE": mape, "MAD": mad, "MSD": msd}


def arima_forecast(series, steps=50, order=(1, 1, 1)):
    model = ARIMA(series, order=order)
    fit = model.fit()
    fc = fit.get_forecast(steps=steps)
    yhat = fc.predicted_mean
    conf = fc.conf_int(alpha=0.05)
    return yhat, conf.iloc[:, 0], conf.iloc[:, 1]



def prophet_forecast(series, steps=50):
    if not HAS_PROPHET:
        raise RuntimeError("Prophet chưa cài. Hãy chạy: pip install prophet")

    dfp = pd.DataFrame({"ds": pd.to_datetime(series.index), "y": series.values})
    dfp = dfp.dropna()
    if len(dfp) < 2:
        raise ValueError("Không đủ dữ liệu cho Prophet")

    # 👇 Quan trọng: dùng Prophet, KHÔNG phải tf
    m = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
    )
    m.fit(dfp)

    future = m.make_future_dataframe(periods=steps, freq="B")
    fcst = m.predict(future).set_index("ds")

    yhat = fcst["yhat"].iloc[-steps:]
    low  = fcst["yhat_lower"].iloc[-steps:]
    up   = fcst["yhat_upper"].iloc[-steps:]
    return yhat, low, up



def future_index(last_time, steps=50):
    return pd.bdate_range(start=last_time, periods=steps + 1, freq="B")[1:]


def backtest_and_metrics(series, model_kind="ARIMA", steps=20):
    """fit on train (=all-steps), forecast 'steps', then score on test"""
    train, test = series.iloc[:-steps], series.iloc[-steps:]

    if model_kind == "ARIMA":
        yhat, low, up = arima_forecast(train, steps=steps)
    elif model_kind == "Prophet":
        try:
            yhat, low, up = prophet_forecast(train, steps=steps)
        except ValueError:
            # fallback ARIMA nếu dữ liệu quá ít
            yhat, low, up = arima_forecast(train, steps=steps)
    elif model_kind == "LSTM":
        yhat, low, up = lstm_forecast(train, steps=steps)
    elif model_kind == "Transformer":
        yhat, low, up = transformer_forecast(train, steps=steps)
    else:   # Moving Average fallback
        yhat = pd.Series([train.tail(20).mean()] * steps, index=test.index)
        sd = train.std()
        low = yhat - 1.96 * sd
        up  = yhat + 1.96 * sd

    metrics = compute_metrics(test.values, yhat.values)
    return yhat, low, up, metrics
