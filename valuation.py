# valuation.py
import numpy as np
from vnstock import Vnstock

def dcf_valuation(eps, payout, growth, growth_term, discount, years=5):
    """
    eps: EPS hiện tại
    payout: payout ratio (tỷ lệ chia cổ tức tiền mặt)
    growth: tăng trưởng EPS 5 năm đầu
    growth_term: tăng trưởng dài hạn sau đó
    discount: required return
    """
    eps_t = eps
    cash_flows = []

    for t in range(1, years+1):
        eps_t *= (1 + growth)
        div_t = eps_t * payout
        cash_flows.append(div_t / (1 + discount)**t)

    # terminal value theo Gordon
    eps_T1 = eps_t * (1 + growth_term)
    div_T1 = eps_T1 * payout
    tv = div_T1 / (discount - growth_term)
    pv_tv = tv / (1 + discount)**years

    fair_value = sum(cash_flows) + pv_tv
    return fair_value

def load_eps_payout(symbol: str):
    """
    Lấy EPS và payout ratio gần nhất từ vnstock.
    Trả về (eps, payout) với payout dạng 0–1. Có fallback khi thiếu dữ liệu.
    """
    stock = Vnstock().stock(symbol=symbol, source=["VCI", "TCBS", "SSI"] )

    eps_val = np.nan
    payout = np.nan

    try:
        ratio = stock.finance.ratio(period="year", last=1)   # bảng chỉ số tài chính năm gần nhất

        # kiếm cột có chữ 'eps'
        eps_cols = [c for c in ratio.columns if "eps" in c.lower()]
        if eps_cols:
            eps_val = float(ratio[eps_cols[0]].iloc[0])

        # kiếm cột có chữ 'payout'
        pay_cols = [c for c in ratio.columns if "payout" in c.lower()]
        if pay_cols:
            payout_raw = float(ratio[pay_cols[0]].iloc[0])
            # thường payout trong bảng là %, nên chia 100
            payout = payout_raw / 100.0
    except Exception:
        pass

    # fallback nếu không lấy được
    if np.isnan(eps_val) or eps_val <= 0:
        eps_val = 3000.0
    if np.isnan(payout) or payout <= 0 or payout > 1:
        payout = 0.4

    return eps_val, payout
