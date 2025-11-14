import pandas as pd
from vnstock import Vnstock
import ta

symbol = "HPG"
start = "2025-06-01"
end   = "2025-07-31"

# 1. Lấy dữ liệu giá lịch sử
stock = Vnstock().stock(symbol=symbol, source="VCI")
df = stock.quote.history(start=start, end=end, interval='1D')

# Đảm bảo cột tên đúng: 'high', 'low', 'close', 'volume'
df.rename(columns={
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume_match': 'Volume'  # nếu tên volume khác thì đổi cho khớp
}, inplace=True)

# 2. Tính MFI(14)
df['MFI_14'] = ta.volume.MFIIndicator(
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    volume=df['volume'],
    window=14
).money_flow_index()

# 3. Lọc đúng ngày 2025-07-17
df['date'] = pd.to_datetime(df['time'])  # hoặc cột ngày mà vnstock trả về
mfi_2025_07_17 = df.loc[df['date'] == '2025-07-17', 'MFI_14']

print(mfi_2025_07_17)