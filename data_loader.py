from vnstock import Vnstock
from datetime import date
import pandas as pd

def load_stock(symbol: str,
               start: str = "2018-01-01",
               end: str | None = None,
               interval: str = "1D") -> pd.DataFrame:
    if end is None:
        end = date.today().strftime("%Y-%m-%d")

    sources = ["VCI", "TCBS", "SSI"]   # thử lần lượt
    last_err = None

    for src in sources:
        try:
            stock = Vnstock().stock(symbol=symbol, source=src)
            df = stock.quote.history(start=start, end=end, interval=interval)
            if df is not None and len(df) > 0:
                return df   # ✅ thành công
        except Exception as e:
            last_err = e

    # nếu tất cả đều fail
    raise RuntimeError(f"Failed to fetch data from all sources: {last_err}")
