# app.py
import streamlit as st
import pandas as pd
import numpy as np
import ta
from data_loader import load_stock
from forecast import (
    arima_forecast, prophet_forecast, future_index,
    backtest_and_metrics, HAS_TF
)
from indicators import (
    macd_signals_full, rsi_signals,
    stochastic_signals, adx_di_signals,
    golden_cross_signals
)
from signals import evaluate_signals
from utils import plot_price_with_bands, confidence_color
import plotly.graph_objects as go
from patterns import detect_candlestick_patterns
from indicators import money_flow_indicators
from valuation import dcf_valuation, load_eps_payout
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

#Seurity
# def check_password():
#     """Trả về True nếu pass đúng, False nếu sai (hoặc chưa nhập)."""

#     def password_entered():
#         """So sánh pass nhập với pass lưu trong secrets."""
#         if st.session_state["password"] == st.secrets["APP_PASSWORD"]:
#             st.session_state["password_correct"] = True
#             del st.session_state["password"]  # xóa pass khỏi state cho an toàn
#         else:
#             st.session_state["password_correct"] = False

#     # Lần đầu vào app
#     if "password_correct" not in st.session_state:
#         st.text_input(
#             "Nhập mật khẩu để truy cập:",
#             type="password",
#             on_change=password_entered,
#             key="password",
#         )
#         return False

#     # Đã nhập nhưng sai
#     if not st.session_state["password_correct"]:
#         st.text_input(
#             "Sai mật khẩu, nhập lại:",
#             type="password",
#             on_change=password_entered,
#             key="password",
#         )
#         st.error("❌ Mật khẩu không đúng.")
#         return False

#     # Đúng rồi
#     return True


# # ⚠️ Chặn toàn bộ app nếu chưa qua cửa password
# if not check_password():
#     st.stop()


# ============ UI ============
st.set_page_config(page_title="Phân tích cổ phiếu đa khung", layout="wide")
page = st.sidebar.radio(
    "Chọn trang",
    ["📈 Phân tích cổ phiếu", "📊 Thị trường realtime"]
)
if page == "📈 Phân tích cổ phiếu":
    st.title("📈 Phân tích cổ phiếu đa khung thời gian")

    col1, col2, col3 = st.columns([1.2, 1, 1.4])
    with col1:
        symbol = st.text_input("Mã cổ phiếu (HOSE/HNX/UPCOM)", "HPG").upper().strip()
    with col2:
        range_pick = st.selectbox("Khoảng thời gian", ["3M", "6M", "1Y", "All"], index=2)
    with col3:
        model_pick = st.selectbox(
            "Mô hình dự báo",
            ["ARIMA","Prophet","Moving Average (fallback)"],
            index=0,
        )


    if "analyzed" not in st.session_state:
        st.session_state["analyzed"] = False

    if st.button("Phân tích"):
        try:
            df = load_stock(symbol, start="2018-01-01", end=None, interval="1D")
            st.session_state["df"] = df
            st.session_state["symbol"] = symbol
            st.session_state["analyzed"] = True
            st.success("Đã phân tích thành công!")
        except Exception as e:
            st.session_state["analyzed"] = False
            st.error(f"Lỗi: {e}")

    if st.session_state.get("analyzed", False):
        df = st.session_state["df"].copy()
        symbol = st.session_state["symbol"]

        # ---- date filter
        if range_pick != "All":
            months = {"3M": 3, "6M": 6, "1Y": 12}[range_pick]
            start_cut = df["time"].max() - pd.DateOffset(months=months)
            df = df[df["time"] >= start_cut].reset_index(drop=True)

        tab_short, tab_mid, tab_long, tab_adv, tab_fa= st.tabs(
            ["⏱️ Ngắn hạn", "📆 Trung hạn", "🏦 Dài hạn", "🔬 Tín hiệu nâng cao", "📊Định giá cp"]
        )

        # ================= NGẮN HẠN =================
        with tab_short:
            st.subheader("EMA, MACD, RSI, Stochastic, Bollinger + Forecast")

            df["EMA20"] = ta.trend.ema_indicator(df["close"], window=20)
            macd_buys, macd_sells, macd, macd_sig = macd_signals_full(df)
            rsi = ta.momentum.rsi(df["close"], window=14)
            bb = ta.volatility.BollingerBands(df["close"])
            bb_high, bb_low = bb.bollinger_hband(), bb.bollinger_lband()

            series = pd.Series(df["close"].values, index=df["time"])
            steps = 50

            try:
                if model_pick == "Prophet" and not HAS_TF:
                    raise RuntimeError("Prophet not installed")

                if model_pick == "Prophet":
                    yhat, low, up = prophet_forecast(series, steps)
                elif model_pick == "ARIMA":
                    yhat, low, up = arima_forecast(series, steps)
                else:
                    fc_idx = future_index(series.index[-1], steps)
                    mean_val = series.tail(20).mean()
                    sd = series.std()
                    yhat = pd.Series([mean_val] * steps, index=fc_idx)
                    low  = yhat - 1.96 * sd
                    up   = yhat + 1.96 * sd

                fc_idx = future_index(series.index[-1], steps)
                if not isinstance(yhat, pd.Series):
                    yhat = pd.Series(yhat.values, index=fc_idx)
                    low  = pd.Series(low.values,  index=fc_idx)
                    up   = pd.Series(up.values,   index=fc_idx)
            except Exception:
                fc_idx = future_index(series.index[-1], steps)
                mean_val = series.tail(20).mean()
                sd = series.std()
                yhat = pd.Series([mean_val] * steps, index=fc_idx)
                low  = yhat - 1.96 * sd
                up   = yhat + 1.96 * sd

            _, _, _, m = backtest_and_metrics(
                series,
                model_kind = (
                    "Prophet" if (model_pick=="Prophet" and HAS_TF) else
                    "ARIMA"   if model_pick=="ARIMA" else
                    "MA"
                ),
                steps=20
            )

            fig = plot_price_with_bands(
                df,
                title=f"{symbol} · Ngắn hạn",
                buys=macd_buys,
                sells=macd_sells,
                extra_lines=[
                    {"name": "EMA20", "y": df["EMA20"]},
                    {"name": "BB High", "y": bb_high},
                    {"name": "BB Low", "y": bb_low},
                ],
                fc_idx=fc_idx,
                yhat=yhat,
                low=low,
                up=up,
            )
            st.plotly_chart(fig, use_container_width=True)

            color = confidence_color(m["MAPE"])
            st.markdown(
                f"""
                <div style="padding:8px;border-radius:6px;background-color:{color};color:white;font-weight:bold">
                    🔍 Accuracy (backtest): n={m['n']} | MAPE={m['MAPE']:.2f}% | MAD={m['MAD']:.3f} | MSD={m['MSD']:.3f}
                </div>
                """,
                unsafe_allow_html=True,
            )

            df_trades, perf = evaluate_signals(macd_buys, macd_sells)
            st.subheader("📊 Đánh giá tín hiệu MACD")
            if len(df_trades) > 0:
                st.dataframe(df_trades.tail(10))
                conf_color = (
                    "green" if perf["WinRate"] > 60 else
                    "gold"  if perf["WinRate"] > 40 else
                    "red"
                )
                st.markdown(
                    f"""
                    **Tổng số giao dịch:** {perf['TotalTrades']}  
                    🟢 **Win rate:** {perf['WinRate']:.1f}%  
                    📈 **Avg Gain:** {perf['AvgGain']:.2f}%  
                    🔻 **Avg Loss:** {perf['AvgLoss']:.2f}%  
                    💰 **Expectancy:** {perf['Expectancy']:.2f}%  
                    """
                )
            else:
                st.info("Không đủ tín hiệu MACD để đánh giá.")


            st.caption("Buy/Sell markers theo MACD cross; dải tím là 95% CI của mô hình dự báo đã chọn.")

        with tab_mid:
            st.subheader("SMA50, EMA50, ADX, Ichimoku, SAR + Forecast")
            df['SMA50'] = ta.trend.sma_indicator(df['close'], window=50)
            df['EMA50'] = ta.trend.ema_indicator(df['close'], window=50)
            adx = ta.trend.adx(df['high'], df['low'], df['close'])
            ich = ta.trend.IchimokuIndicator(df['high'], df['low'], window1=9, window2=26, window3=52)
            kumo_up, kumo_low = ich.ichimoku_a(), ich.ichimoku_b()
            sar_up = ta.trend.psar_up(df['high'], df['low'], df['close'])

            series = pd.Series(df['close'].values, index=df['time'])
            steps = 60
            try:
                yhat, low, up = (prophet_forecast(series, steps) if (model_pick=="Prophet" and HAS_TF)
                                else arima_forecast(series, steps) if model_pick=="ARIMA"
                                else (pd.Series([series.tail(50).mean()]*steps, index=future_index(series.index[-1], steps)),
                                    pd.Series([series.tail(50).mean()-1.96*series.std()]*steps, index=future_index(series.index[-1], steps)),
                                    pd.Series([series.tail(50).mean()+1.96*series.std()]*steps, index=future_index(series.index[-1], steps))))
                fc_idx = future_index(series.index[-1], steps)
                if not isinstance(yhat, pd.Series):
                    yhat = pd.Series(yhat.values, index=fc_idx)
                    low  = pd.Series(low.values,  index=fc_idx)
                    up   = pd.Series(up.values,   index=fc_idx)
            except Exception:
                fc_idx = future_index(series.index[-1], steps)
                yhat = pd.Series([series.tail(50).mean()]*steps, index=fc_idx)
                sd = series.std()
                low = yhat - 1.96*sd
                up  = yhat + 1.96*sd

            _, _, _, m = backtest_and_metrics(series, model_kind=("Prophet" if (model_pick=="Prophet" and HAS_TF) else ("ARIMA" if model_pick=="ARIMA" else "MA")), steps=50)

            fig2 = plot_price_with_bands(
                df, title=f"{symbol} · Trung hạn",
                extra_lines=[{"name":"SMA50","y":df['SMA50']},
                            {"name":"EMA50","y":df['EMA50']},
                            {"name":"Ichimoku A","y":kumo_up},
                            {"name":"Ichimoku B","y":kumo_low}],
                fc_idx=fc_idx, yhat=yhat, low=low, up=up
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.write(f"**Accuracy (50 phiên backtest) — n={m['n']} | MAPE={m['MAPE']:.2f}% | MAD={m['MAD']:.3f} | MSD={m['MSD']:.3f}**")
            st.caption("SAR & ADX có thể xem nhanh trong bảng dưới.")
            st.dataframe(pd.DataFrame({"ADX": adx, "SAR_up": sar_up}).tail(10))

        # ----- DÀI HẠN -----
        with tab_long:
            st.subheader("SMA100, SMA200, Volume, Golden Cross + Forecast")
            buys_gc, sells_gc, sma100, sma200 = golden_cross_signals(df, 100, 200)

            series = pd.Series(df['close'].values, index=df['time'])
            steps = 90
            try:
                yhat, low, up = (prophet_forecast(series, steps) if (model_pick=="Prophet" and HAS_TF)
                                else arima_forecast(series, steps) if model_pick=="ARIMA"
                                else (pd.Series([series.tail(200).mean()]*steps, index=future_index(series.index[-1], steps)),
                                    pd.Series([series.tail(200).mean()-1.96*series.std()]*steps, index=future_index(series.index[-1], steps)),
                                    pd.Series([series.tail(200).mean()+1.96*series.std()]*steps, index=future_index(series.index[-1], steps))))
                fc_idx = future_index(series.index[-1], steps)
                if not isinstance(yhat, pd.Series):
                    yhat = pd.Series(yhat.values, index=fc_idx)
                    low  = pd.Series(low.values,  index=fc_idx)
                    up   = pd.Series(up.values,   index=fc_idx)
            except Exception:
                fc_idx = future_index(series.index[-1], steps)
                yhat = pd.Series([series.tail(200).mean()]*steps, index=fc_idx)
                sd = series.std()
                low = yhat - 1.96*sd
                up  = yhat + 1.96*sd

            _, _, _, m = backtest_and_metrics(series, model_kind=("Prophet" if (model_pick=="Prophet" and HAS_TF) else ("ARIMA" if model_pick=="ARIMA" else "MA")), steps=100)

            fig3 = plot_price_with_bands(
                df, title=f"{symbol} · Dài hạn",
                buys=buys_gc, sells=sells_gc,
                extra_lines=[{"name":"SMA100","y":sma100},
                            {"name":"SMA200","y":sma200}],
                fc_idx=fc_idx, yhat=yhat, low=low, up=up
            )
            st.plotly_chart(fig3, use_container_width=True)
            st.write(f"**Accuracy (100 phiên backtest) — n={m['n']} | MAPE={m['MAPE']:.2f}% | MAD={m['MAD']:.3f} | MSD={m['MSD']:.3f}**")
            st.caption("Buy/Sell theo Golden (SMA100 cắt lên SMA200) & Dead Cross.")

            df_trades_gc, perf_gc = evaluate_signals(buys_gc, sells_gc)
            st.subheader("📊 Đánh giá tín hiệu Golden Cross / Dead Cross")
            if len(df_trades_gc) > 0:
                st.dataframe(df_trades_gc.tail(10))
                conf_color = "green" if perf['WinRate'] > 60 else "gold" if perf['WinRate'] > 40 else "red"
                st.markdown(
                    f"""
                    <div style="background-color:{conf_color};padding:6px;border-radius:6px;color:white;font-weight:bold">
                        ⚙️ Win rate: {perf['WinRate']:.1f}% | Avg Gain: {perf['AvgGain']:.2f}% | Avg Loss: {perf['AvgLoss']:.2f}% | Expectancy: {perf['Expectancy']:.2f}%
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"""
                    **Tổng số giao dịch:** {perf_gc['TotalTrades']}  
                    🟢 **Win rate:** {perf_gc['WinRate']:.1f}%  
                    📈 **Avg Gain:** {perf_gc['AvgGain']:.2f}%  
                    🔻 **Avg Loss:** {perf_gc['AvgLoss']:.2f}%  
                    💰 **Expectancy:** {perf_gc['Expectancy']:.2f}%  
                    """
                )
            else:
                st.info("Không đủ tín hiệu Golden Cross để đánh giá.")

                #Nâng cao
        with tab_adv:
            st.subheader(f"🔬 Tín hiệu nâng cao cho {symbol}")

            # Đảm bảo df có cột time, close, high, low, volume
            df_sig = (
                df.copy()
                .dropna(subset=['close', 'high', 'low', 'volume'])  # 👈 thêm volume
                .reset_index(drop=True)
            )


            # 1) MACD
            macd_buys, macd_sells, macd_line, macd_sig = macd_signals_full(df_sig)
            macd_trades, macd_perf = evaluate_signals(macd_buys, macd_sells)
            macd_perf["Strategy"] = "MACD Cross"

            # 2) RSI 30/70
            rsi_buys, rsi_sells, rsi = rsi_signals(df_sig)
            rsi_trades, rsi_perf = evaluate_signals(rsi_buys, rsi_sells)
            rsi_perf["Strategy"] = "RSI 30/70 Cross"

            # 3) Stochastic
            sto_buys, sto_sells, sto_k, sto_d = stochastic_signals(df_sig)
            sto_trades, sto_perf = evaluate_signals(sto_buys, sto_sells)
            sto_perf["Strategy"] = "Stochastic Cross (20/80)"

            # 4) ADX + DI Cross
            adx_buys, adx_sells, adx_val, di_pos, di_neg = adx_di_signals(df_sig)
            adx_trades, adx_perf = evaluate_signals(adx_buys, adx_sells)
            adx_perf["Strategy"] = "ADX + DI Cross"

            # 5) Golden Cross / Death Cross
            gc_buys, gc_sells, sma_s, sma_l = golden_cross_signals(df_sig, 50, 200)
            gc_trades, gc_perf = evaluate_signals(gc_buys, gc_sells)
            gc_perf["Strategy"] = "Golden/Death Cross (SMA50/200)"

            # === Bảng so sánh tổng hợp ===
            perf_list = [macd_perf, rsi_perf, sto_perf, adx_perf, gc_perf]
            perf_df = pd.DataFrame(perf_list)[
                ["Strategy", "TotalTrades", "WinRate", "AvgGain", "AvgLoss", "Expectancy"]
            ]

            st.markdown("### 📋 So sánh hiệu quả các chiến lược tín hiệu")
            st.dataframe(perf_df.style.format({
                "WinRate": "{:.1f}%",
                "AvgGain": "{:.2f}%",
                "AvgLoss": "{:.2f}%",
                "Expectancy": "{:.2f}%"
            }))

            # === Chọn chiến lược để vẽ chart Buy/Sell ===
            st.markdown("### 📈 Minh hoạ tín hiệu trên biểu đồ giá")
            strategy_choice = st.selectbox(
                "Chọn chiến lược để hiển thị tín hiệu:",
                ["MACD Cross", "RSI 30/70 Cross", "Stochastic Cross (20/80)", "ADX + DI Cross", "Golden/Death Cross (SMA50/200)"]
            )

            if strategy_choice == "MACD Cross":
                buys, sells = macd_buys, macd_sells
                title = f"{symbol} · MACD Buy/Sell signals"
            elif strategy_choice == "RSI 30/70 Cross":
                buys, sells = rsi_buys, rsi_sells
                title = f"{symbol} · RSI 30/70 Buy/Sell signals"
            elif strategy_choice == "Stochastic Cross (20/80)":
                buys, sells = sto_buys, sto_sells
                title = f"{symbol} · Stochastic Buy/Sell signals"
            elif strategy_choice == "ADX + DI Cross":
                buys, sells = adx_buys, adx_sells
                title = f"{symbol} · ADX + DI Buy/Sell signals"
            else:
                buys, sells = gc_buys, gc_sells
                title = f"{symbol} · Golden/Death Cross signals"

            fig_sig = go.Figure()
            fig_sig.add_trace(go.Scatter(x=df_sig['time'], y=df_sig['close'],
                                        name="Close", mode="lines", line=dict(color="steelblue")))
            if buys is not None and len(buys) > 0:
                fig_sig.add_trace(go.Scatter(x=buys['time'], y=buys['close'],
                                            mode="markers", name="Buy",
                                            marker=dict(symbol="triangle-up", size=12, color="green")))
            if sells is not None and len(sells) > 0:
                fig_sig.add_trace(go.Scatter(x=sells['time'], y=sells['close'],
                                            mode="markers", name="Sell",
                                            marker=dict(symbol="triangle-down", size=12, color="red")))
            fig_sig.update_layout(title=title, height=520, legend=dict(orientation="h"))
            st.plotly_chart(fig_sig, use_container_width=True)

            # === (Tuỳ chọn) Hiển thị chi tiết các lệnh của chiến lược được chọn ===
            st.markdown("### 📑 Các lệnh giao dịch theo chiến lược đã chọn")
            if strategy_choice == "MACD Cross":
                trades = macd_trades
            elif strategy_choice == "RSI 30/70 Cross":
                trades = rsi_trades
            elif strategy_choice == "Stochastic Cross (20/80)":
                trades = sto_trades
            elif strategy_choice == "ADX + DI Cross":
                trades = adx_trades
            else:
                trades = gc_trades

            if trades is not None and len(trades) > 0:
                st.dataframe(trades.tail(20))
            else:
                st.info("Chưa có đủ tín hiệu để tạo giao dịch cho chiến lược này.")

            # patterns = detect_candlestick_patterns(df_sig)
            # st.markdown("### Mô hình nến gần đây")
            # st.dataframe(patterns.tail(30))      
            # show_pattern = st.selectbox(
            #     "Hiển thị mô hình nến:",
            #     ["None","Doji","Hammer","Shooting star","Bullish Engulfing","Bearish Engulfing"]
            # )

            # mask = None
            # if show_pattern == "Doji":
            #     mask = patterns["doji"]
            # elif show_pattern == "Hammer":
            #     mask = patterns["hammer"]
            # ...

            # if mask is not None:
            #     pts = df_sig[mask]
            #     fig_sig.add_trace(go.Scatter(
            #         x=pts["time"], y=pts["close"],
            #         mode="markers", name=show_pattern,
            #         marker=dict(symbol="x", size=12, color="orange")
            #     ))
            mfi, obv = money_flow_indicators(df_sig)
            rsi_14 = ta.momentum.rsi(df_sig["close"], window=14)
            df_mf = pd.DataFrame({
                "time": df_sig["time"],
                "Close": df_sig["close"],
                "MFI(14)": mfi,
                "RSI(14)": rsi_14,
                "OBV": obv
            }).set_index("time")
            df_nf = pd.DataFrame({
                "time": df_sig["time"],
                "OBV": obv
            }).set_index("time")


            st.markdown("### 🔄 Phân tích dòng tiền (MFI / OBV)")
            st.line_chart(df_nf.tail(200))
            st.caption("- MFI > 80: vùng quá mua, < 20: quá bán\n"
                    "- RSI > 70: vùng quá mua, < 30: quá bán\n"
                    "- OBV tăng cùng giá → dòng tiền ủng hộ xu hướng; OBV đi ngược giá → cảnh báo phân kỳ.")
            # để debug xem có số hay không
            n = st.slider("Số phiên gần nhất để phân tích MFI/RSI/giá", 20, 250, 100, 10)
            subset = df_mf.tail(n)
            st.dataframe(subset)
            st.markdown("### 📈 Biểu đồ MFI(14), RSI(14) & Giá")

            # 🔘 Chọn đường muốn hiển thị
            col_l, col_r = st.columns(2)
            with col_l:
                show_price = st.checkbox("Hiển thị Giá", value=True)
                show_mfi   = st.checkbox("Hiển thị MFI(14)", value=True)
            with col_r:
                show_rsi   = st.checkbox("Hiển thị RSI(14)", value=True)

            fig_mf = make_subplots(specs=[[{"secondary_y": True}]])
            x = subset.index

            # Giá (trục y phụ)
            if show_price:
                fig_mf.add_trace(
                    go.Scatter(x=x, y=subset["Close"], name="Giá đóng cửa", mode="lines"),
                    secondary_y=True,
                )
            # MFI (trục y chính)
            if show_mfi:
                fig_mf.add_trace(
                    go.Scatter(x=x, y=subset["MFI(14)"], name="MFI(14)", mode="lines"),
                    secondary_y=False,
                )
                # RSI (trục y chính)
            if show_rsi:
                fig_mf.add_trace(
                    go.Scatter(x=x, y=subset["RSI(14)"], name="RSI(14)", mode="lines"),
                    secondary_y=False,
                )   

            # Ngưỡng MFI 80 / 20
            fig_mf.add_hline(y=80, line_dash="dash", line_color="red",
                            annotation_text="MFI 80", annotation_position="top left")
            fig_mf.add_hline(y=20, line_dash="dash", line_color="green",
                            annotation_text="MFI 20", annotation_position="bottom left")

            # Ngưỡng RSI 70 / 30 (cùng trục 0–100)
            fig_mf.add_hline(y=70, line_dash="dot", line_color="orange",
                            annotation_text="RSI 70", annotation_position="top right")
            fig_mf.add_hline(y=30, line_dash="dot", line_color="blue",
                            annotation_text="RSI 30", annotation_position="bottom right")

            # Setup trục
            fig_mf.update_yaxes(title_text="MFI / RSI (0–100)", range=[0, 100],
                                secondary_y=False)
            fig_mf.update_yaxes(title_text="Giá", secondary_y=True)

            fig_mf.update_layout(
                height=500,
                legend=dict(orientation="h"),
                title=f"{symbol} · MFI(14), RSI(14) & Giá (last {n} bars)",
            )

            st.plotly_chart(fig_mf, use_container_width=True)
            
        with tab_fa:
            st.subheader(f"📊 Định giá chiết khấu cổ tức cho {symbol}")

            # Nút load từ vnstock
            if st.button("🔄 Load EPS & payout từ vnstock"):
                eps_loaded, payout_loaded = load_eps_payout(symbol)
                st.session_state["fa_eps"] = eps_loaded
                st.session_state["fa_payout"] = payout_loaded
                st.success(f"Đã load: EPS ≈ {eps_loaded:,.0f} VND/cp, payout ≈ {payout_loaded:.2f}")

            # Giá trị mặc định cho widget (nếu chưa load thì dùng số cũ)
            eps_default = st.session_state.get("fa_eps", 3000.0)
            payout_default = st.session_state.get("fa_payout", 0.4)

            col1, col2 = st.columns(2)
            with col1:
                current_price = st.number_input(
                    "Giá thị trường hiện tại",
                    min_value=0.0,
                    value=float(df["close"].iloc[-1])
                )
                eps = st.number_input(
                    "EPS 12T gần nhất (VND/cp)",
                    min_value=0.0,
                    value=float(eps_default)
                )
                payout = st.slider(
                    "Payout ratio (tỷ lệ chia cổ tức)",
                    0.0, 1.0, float(payout_default), 0.05
                )

            with col2:
                growth = st.slider("Tăng trưởng EPS 5 năm tới (%)",
                                -10.0, 40.0, 10.0, 1.0) / 100
                growth_term = st.slider("Tăng trưởng dài hạn (%)",
                                        0.0, 8.0, 3.0, 0.5) / 100
                discount = st.slider("Tỷ suất sinh lời yêu cầu r (%)",
                                    8.0, 20.0, 13.0, 0.5) / 100
                years = st.slider("Số năm dự báo chi tiết", 3, 10, 5)

            if st.button("Tính giá trị hợp lý (DCF)"):
                fair = dcf_valuation(eps, payout, growth, growth_term, discount, years)
                mos = (fair - current_price) / current_price * 100
                st.markdown(f"""
                **Giá trị hợp lý ước tính:** `{fair:,.0f} VND/cp`  
                **Margin of safety:** `{mos:,.1f}%`  
                """)

# ==== PAGE 2: Thị trường realtime ====
if page == "📊 Thị trường realtime":
    refresh_sec = st.sidebar.slider(
        "Chu kỳ làm mới bảng realtime (giây)",
        min_value=5, max_value=60, value=60, step=5
    )

    # 🔁 Tự rerun theo chu kỳ (chỉ áp dụng cho page này)
    st_autorefresh(interval=refresh_sec * 1000, key="market_refresh")
    st.title("📊 Thị trường realtime (VNIndex & Watchlist)")

    # 1) VNIndex
    st.subheader("VNIndex (daily)")
    try:
        df_vni = load_stock("VNINDEX", start="2024-01-01")
        df_vni = df_vni.sort_values("time")
        st.line_chart(
            df_vni.set_index("time")["close"],
            height=250
        )

        last = df_vni.iloc[-1]
        prev = df_vni.iloc[-2] if len(df_vni) > 1 else last
        chg = last["close"] - prev["close"]
        pct = chg / prev["close"] * 100 if prev["close"] != 0 else 0

        # Màu cho VNIndex: tăng = xanh, giảm = đỏ, đứng im = xám
        idx_color = "green" if pct > 0 else "red" if pct < 0 else "gray"

        st.markdown(
            f"""
            <div style="font-size:18px;">
                <b>{last['time'].date()}</b> ·
                VNIndex:
                <span style="color:{idx_color};font-weight:bold;">
                    {last['close']:.2f} điểm ({chg:+.2f} | {pct:+.2f}%)
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"Không lấy được dữ liệu VNINDEX: {e}")

    st.markdown("---")

    # 2) Bảng watchlist
    st.subheader("Watchlist cổ phiếu")

    default_list = "HPG, SSI, VCB, VNM, FPT, CMC, HSG, PVO, VND"
    symbols_text = st.text_input(
        "Danh sách mã (phân cách bằng dấu phẩy):",
        value=default_list
    )
    watchlist = [s.strip().upper() for s in symbols_text.split(",") if s.strip()]

    rows = []
    for sym in watchlist:
        try:
            df_sym = load_stock(sym, start="2024-10-01")
            df_sym = df_sym.sort_values("time")
            last = df_sym.iloc[-1]
            prev = df_sym.iloc[-2] if len(df_sym) > 1 else last

            chg = last["close"] - prev["close"]
            pct = chg / prev["close"] * 100 if prev["close"] != 0 else 0

            # Đánh dấu trạng thái để tô màu:
            # tím = trần (giả định %>=6.8), xanh = tăng, đỏ = giảm, vàng = đứng im
            if pct >= 6.8:
                state = "TRẦN"
            elif pct > 0:
                state = "TĂNG"
            elif pct < 0:
                state = "GIẢM"
            else:
                state = "ĐỨNG IM"

            rows.append({
                "Mã": sym,
                "Ngày": last["time"].date(),
                "Giá đóng cửa": last["close"],
                "Thay đổi": chg,
                "% thay đổi": pct,
                "Khối lượng": last.get("volume", None),
                "Trạng thái": state,
            })
        except Exception:
            rows.append({
                "Mã": sym,
                "Ngày": None,
                "Giá đóng cửa": None,
                "Thay đổi": None,
                "% thay đổi": None,
                "Khối lượng": None,
                "Trạng thái": "N/A",
            })

    if rows:
        df_board = pd.DataFrame(rows)
        df_board = df_board.sort_values("% thay đổi", ascending=False)

        # Hàm tô màu từng hàng
        def color_row(row):
            s = row["Trạng thái"]
            if s == "TRẦN":
                color = "#E9D5FF"  # tím nhạt
            elif s == "TĂNG":
                color = "#BBF7D0"  # xanh lá nhạt
            elif s == "GIẢM":
                color = "#FECACA"  # đỏ nhạt
            elif s == "ĐỨNG IM":
                color = "#FEF9C3"  # vàng nhạt
            else:
                return [""] * len(row)
            return [f"background-color: {color};"] * len(row)

        styler = (
            df_board.style
            .apply(color_row, axis=1)
            .format({
                "Giá đóng cửa": "{:,.2f}",
                "Thay đổi": "{:+.2f}",
                "% thay đổi": "{:+.2f}%",
                "Khối lượng": "{:,.0f}",
            })
        )

        st.dataframe(styler, use_container_width=True)
    else:
        st.info("Nhập ít nhất 1 mã để theo dõi.")



                        