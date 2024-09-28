import yfinance as yf
import pandas_ta as ta

def calculate_macd(stock_name, start_date, end_date):
    data = yf.download(stock_name, start=start_date, end=end_date)
    macd = ta.macd(data['Close'])
    
    # Kolon isimlerini kontrol et
    print("MACD Columns:", macd.columns.tolist())
    
    # Veriyi birleştirme ve eksik sütunları kontrol etme
    data = data.join(macd)
    expected_macd_columns = ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']
    for column in expected_macd_columns:
        if column not in data.columns:
            print(f"Missing column: {column}")
    
    return data[expected_macd_columns].dropna()

# Test etmek için örnek çağrı
data = calculate_macd('AAPL', '2023-01-01', '2024-01-01')
print(data.head())
