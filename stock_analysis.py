# stock_analysis.py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import uuid
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import pandas_ta as ta
import csv
from io import StringIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow import keras
import tensorflow as tf

from tensorflow.python.keras.engine.sequential import Sequential

from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D




def fetch_historical_data(stock_name, start_date, end_date):
    data = yf.download(stock_name, start=start_date, end=end_date)
    return data

def get_stock_info(stock_name):
    stock = yf.Ticker(stock_name)
    return stock.info

def get_recent_data(stock_name):
    recent_data = yf.download(stock_name, period="5d")
    return recent_data

def fetch_multiple_stocks_data(stock_list, start_date, end_date):
    multi_data = yf.download(stock_list, start=start_date, end=end_date)
    return multi_data


def show_adjusted_close_prices(stock_name, start_date, end_date):
    data = yf.download(stock_name, start=start_date, end=end_date, auto_adjust=True)
    return data['Close'].to_frame()  # Series'i DataFrame'e dönüştür

def fetch_weekly_data(stock_name, start_date, end_date):
    weekly_data = yf.download(stock_name, start=start_date, end=end_date, interval="1wk")
    return weekly_data

def access_dividends_splits(stock_name):
    stock = yf.Ticker(stock_name)
    dividends = stock.dividends
    splits = stock.splits
    return dividends.to_frame(), splits.to_frame() # bunlar değişti en son

def fetch_latest_stock_price(stock_name):
    stock = yf.Ticker(stock_name)
    return stock.history(period="1d")


def fetch_data_for_specific_stock(stock_name, start_date, end_date):
    data = yf.download(stock_name, start=start_date, end=end_date)
    return data

def fetch_multiple_stocks_over_date_range(stock_list, start_date, end_date):
    data = pd.DataFrame(columns=stock_list)
    for ticker in stock_list:
        data[ticker] = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
    return data

def plot_stock_prices(stock_name, start_date, end_date):
    unique_filename = f'{uuid.uuid4().hex}.png'
    file_path = os.path.join('static', unique_filename)
    
    # Download stock data
    stock_data = yf.download(stock_name, start=start_date, end=end_date)
    
    if stock_data.empty:
        raise ValueError(f"No data found for {stock_name} between {start_date} and {end_date}")
    
    # Plot the stock prices
    plt.figure(figsize=(10, 6))
    plt.plot(stock_data.index, stock_data['Close'], label='Close Price')
    plt.title(f'Stock Prices for {stock_name}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    # Save the plot to a file
    plt.savefig(file_path)
    plt.close()
    
    return unique_filename

def plot_close_prices_multiple_stocks(stock_list, start_date, end_date):
    fig = go.Figure()

    for stock_name in stock_list:
        # Her bir hisse senedinin verisini al
        stock_data = yf.download(stock_name, start=start_date, end=end_date)
        
        if stock_data.empty:
            continue
        
        # Kapanış fiyatlarını ekle
        fig.add_trace(go.Scatter(
            x=stock_data.index,
            y=stock_data['Close'],
            mode='lines',
            name=stock_name
        ))

    # Grafik özelliklerini güncelle
    fig.update_layout(
        title='Close Prices of Multiple Stocks',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white'  # Arka planı beyaz yapmak için tema ayarı
    )

    # Grafik dosyasının yolunu oluştur
    unique_filename = f'{uuid.uuid4().hex}.html'
    file_path = os.path.join('static', unique_filename)

    # Grafik HTML dosyasını kaydet
    fig.write_html(file_path)

    return unique_filename

#YENİ EKLENEN KODLAR

def plot_stock_prices_interactive(stock_name, start_date, end_date):
    # Veriyi al
    stock_data = yf.download(stock_name, start=start_date, end=end_date)
    
    if stock_data.empty:
        raise ValueError(f"No data found for {stock_name} between {start_date} and {end_date}")

    # Grafik oluştur
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=stock_data.index,
        y=stock_data['Close'],
        mode='lines',
        name='Close Price'
    ))

    fig.update_layout(
        title=f'Stock Prices for {stock_name}',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white'
    )

    # Grafik dosyasının yolunu oluştur
    unique_filename = f'{uuid.uuid4().hex}.html'
    file_path = os.path.join('static', unique_filename)

    # Grafik HTML dosyasını kaydet
    fig.write_html(file_path)

    return unique_filename

def calculate_rsi(stock_name, start_date, end_date):
    data = yf.download(stock_name, start=start_date, end=end_date)
    data['RSI'] = ta.rsi(data['Close'])
    return data[['RSI']].dropna()


def calculate_macd(stock_name, start_date, end_date):
    data = yf.download(stock_name, start=start_date, end=end_date)
    
    if data.empty:
        raise ValueError("Veri seti boş. Lütfen geçerli bir hisse senedi ve tarih aralığı girin.")
    
    macd = ta.macd(data['Close'])
    
    if macd is None or macd.empty:
        raise ValueError("MACD hesaplaması başarısız oldu. Lütfen veri setinizi kontrol edin.")
    
    # Kolon isimlerini kontrol et
    print("MACD Columns:", macd.columns.tolist())
    
    # Veriyi birleştirme ve eksik sütunları kontrol etme
    data = data.join(macd)
    expected_macd_columns = ['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']
    for column in expected_macd_columns:
        if column not in data.columns:
            print(f"Missing column: {column}")
    
    return data[expected_macd_columns].dropna()


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


def download_csv(data, filename):
    csv_path = os.path.join('static', filename)
    data.to_csv(csv_path)
    return csv_path

def download_pdf(data, filename):
    html_content = data.to_html()
    pdf_path = os.path.join('static', filename)
    from weasyprint import HTML
    HTML(string=html_content).write_pdf(pdf_path)
    return pdf_path

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_model(stock_name, start_date, end_date):
    data = yf.download(stock_name, start=start_date, end=end_date)
    data = data[['Close']].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    train_data_len = int(np.ceil(0.8 * len(scaled_data)))
    train_data = scaled_data[:train_data_len]
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = create_lstm_model((x_train.shape[1], 1))
    model.fit(x_train, y_train, epochs=50, batch_size=1, verbose=2)

    return model, scaler

def predict_future_prices(model, scaler, stock_name, start_date, end_date):
    data = yf.download(stock_name, start=start_date, end=end_date)
    data = data[['Close']].values
    scaled_data = scaler.transform(data)

    x_test = []
    for i in range(60, len(scaled_data)):
        x_test.append(scaled_data[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    return predictions


def plot_candlestick_chart(stock_name, start_date, end_date):
    stock_data = yf.download(stock_name, start=start_date, end=end_date)
    
    if stock_data.empty:
        raise ValueError(f"No data found for {stock_name} between {start_date} and {end_date}")
    
    fig = go.Figure(data=[go.Candlestick(
        x=stock_data.index,
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close']
    )])
    
    fig.update_layout(
        title=f'Candlestick Chart for {stock_name}',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white'
    )
    
    unique_filename = f'{uuid.uuid4().hex}.html'
    file_path = os.path.join('static', unique_filename)
    
    if not os.path.exists('static'):
        os.makedirs('static')
    
    fig.write_html(file_path)
    
    return unique_filename