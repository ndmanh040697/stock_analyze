# data_loader.py
from vnstock import Vnstock
from datetime import date
import pandas as pd

def load_stock(symbol: str,
               start: str = "2018-01-01",
               end: str | None = None,
               interval: str = "1D") -> pd.DataFrame:
    """Tải dữ liệu lịch sử cho 1 mã cổ phiếu."""
    if end is None:
        end = date.today().strftime("%Y-%m-%d")

    stock = Vnstock().stock(symbol=symbol, source="VCI")
    df = stock.quote.history(start=start, end=end, interval=interval)
    return df
