# --- start_app.ps1 ---
# Tự động chuyển sang ổ và thư mục dự án, kích hoạt venv, rồi chạy app Python

Set-Location F:\Production\myproject\stock_analyze
streamlit run app.py

# Giữ cửa sổ PowerShell mở sau khi chạy xong (để xem log nếu có lỗi)
Pause