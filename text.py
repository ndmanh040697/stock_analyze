from vnstock import Trading
t = Trading(source="vci")
df_raw = t.foreign_trade(...)
print(df_raw.head(), df_raw.columns)