import yfinance as yf
import pandas_ta as ta

def calculate_bollinger_bands(stock_name, start_date, end_date):
    data = yf.download(stock_name, start=start_date, end=end_date)
    bbands = ta.bbands(data['Close'])
    
    # Kolon isimlerini kontrol et
    print("Bollinger Bands Columns:", bbands.columns.tolist())
    
    # Veriyi birleştirme ve eksik sütunları kontrol etme
    data = data.join(bbands)
    expected_bbands_columns = ['BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0']
    for column in expected_bbands_columns:
        if column not in data.columns:
            print(f"Missing column: {column}")
    
    return data[expected_bbands_columns].dropna()

# Test etmek için örnek çağrı
data = calculate_bollinger_bands('AAPL', '2023-01-01', '2024-01-01')
print(data.head())
