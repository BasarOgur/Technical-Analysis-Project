import base64
from curses import flash
from io import BytesIO, StringIO
import io
from flask import Flask, redirect, render_template, request, send_file, send_from_directory, url_for
from fpdf import FPDF
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
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

from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense

from sklearn.preprocessing import MinMaxScaler



from stock_analysis import (
    fetch_historical_data, get_stock_info, get_recent_data, 
    fetch_multiple_stocks_data, show_adjusted_close_prices, 
    fetch_weekly_data, access_dividends_splits, fetch_latest_stock_price, 
    plot_stock_prices, fetch_data_for_specific_stock, 
    fetch_multiple_stocks_over_date_range, plot_close_prices_multiple_stocks, plot_stock_prices_interactive, 
    calculate_rsi, calculate_macd, calculate_bollinger_bands, download_csv, download_pdf, create_lstm_model, train_lstm_model, predict_future_prices, plot_candlestick_chart)


app = Flask(__name__)

# Static dosyalar için dizini ayarla
app.config['UPLOAD_FOLDER'] = 'static/'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/fetch', methods=['POST'])
def fetch():
    stock_name = request.form['stock_name']
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    choice = int(request.form['choice'])
    
    # Hisseleri virgüllerle ayır
    stock_list = [name.strip() for name in stock_name.split(',')]
    
    if choice == 1:
        data_html = fetch_historical_data(stock_name, start_date, end_date).to_html()
        return render_template('result.html', tables=[data_html], titles=['Historical Data'], choice=choice)
    elif choice == 2:
        data = get_stock_info(stock_name)
        return render_template('result.html', tables=[str(data)], titles=['Stock Info'], choice=choice)
    elif choice == 3:
        data = get_recent_data(stock_name)
        return render_template('result.html', tables=[data.to_html()], titles=['Recent Data'], choice=choice)
    elif choice == 4:
        data = fetch_multiple_stocks_data(stock_list, start_date, end_date)
        return render_template('result.html', tables=[data.to_html()], titles=['Multiple Stocks Data'], choice=choice)
    elif choice == 5:
        data = show_adjusted_close_prices(stock_name, start_date, end_date)
        return render_template('result.html', tables=[data.to_html()], titles=['Adjusted Close Prices'], choice=choice)
    elif choice == 6:
        data = fetch_weekly_data(stock_name, start_date, end_date)
        return render_template('result.html', tables=[data.to_html()], titles=['Weekly Data'], choice=choice)
    elif choice == 7:
        dividends, splits = access_dividends_splits(stock_name)
        return render_template('result.html', tables=[dividends.to_html() + splits.to_html()], titles=['Dividends and Splits'], choice=choice)
    elif choice == 8:
        data = fetch_latest_stock_price(stock_name)
        return render_template('result.html', tables=[data.to_html()], titles=['Latest Stock Price'], choice=choice)
    elif choice == 9:
        image_filename = plot_stock_prices_interactive(stock_name, start_date, end_date)
        return render_template('result.html', image=image_filename, titles=['Stock Prices Plot'], choice=choice)
    elif choice == 10:
        data = fetch_data_for_specific_stock(stock_name, start_date, end_date)
        return render_template('result.html', tables=[data.to_html()], titles=['Specific Stock Data'], choice=choice)
    elif choice == 11:
        data = fetch_multiple_stocks_over_date_range(stock_list, start_date, end_date)
        return render_template('result.html', tables=[data.to_html()], titles=['Multiple Stocks Over Date Range'], choice=choice)
    elif choice == 12:
        image_filename = plot_close_prices_multiple_stocks(stock_list, start_date, end_date)
        return render_template('result.html', image=image_filename, titles=['Multiple Stocks Close Prices Plot'], choice=choice)
    elif choice == 13:  # RSI
        data = calculate_rsi(stock_name, start_date, end_date)
        return render_template('result.html', tables=[data.to_html()], titles=['RSI'], choice=choice)
    elif choice == 14:  # MACD
        data = calculate_macd(stock_name, start_date, end_date)
        return render_template('result.html', tables=[data.to_html()], titles=['MACD'], choice=choice)
    elif choice == 15:  # Bollinger Bands
        data = calculate_bollinger_bands(stock_name, start_date, end_date)
        return render_template('result.html', tables=[data.to_html()], titles=['Bollinger Bands'], choice=choice)
    elif choice == 16:  # Stock Price Forecast Results
        model, scaler = train_lstm_model(stock_name, start_date, end_date)
        predictions = predict_future_prices(model, scaler, stock_name, start_date, end_date)

        # Plot predictions
        plt.figure(figsize=(12, 6))
        plt.plot(predictions, label='Predicted Prices')
        plt.title(f'Stock Price Forecast for {stock_name}')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        
        # Save the plot to a BytesIO object and encode it as base64
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        img_base64 = base64.b64encode(img.getvalue()).decode()
        
        return render_template('result.html', titles=['Stock Price Forecast Results'], image=f'data:image/png;base64,{img_base64}', choice=choice)
    elif choice == 17:  # Candlestick Chart
        candlestick_chart_path = plot_candlestick_chart(stock_name, start_date, end_date)
        return render_template('result.html', html_file=candlestick_chart_path, titles=['Candlestick Chart'], choice=choice)
    else:
        return "Invalid choice"

@app.route('/process_stock_data', methods=['POST'])
def process_stock_data():
    stock_names = request.form['stock_names']
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    choice = int(request.form['choice'])  # choice değeri alınır ve integer'a dönüştürülür
    
    stock_list = [name.strip() for name in stock_names.split(',')]
    
    if choice == 4:
        data = fetch_multiple_stocks_data(stock_list, start_date, end_date)
        return render_template('result.html', tables=[data.to_html()], titles=['Multiple Stocks Data'], choice=choice)
    elif choice == 11:
        data = fetch_multiple_stocks_over_date_range(stock_list, start_date, end_date)
        return render_template('result.html', tables=[data.to_html()], titles=['Multiple Stocks Over Date Range'], choice=choice)
    elif choice == 12:
        image_filename = plot_close_prices_multiple_stocks(stock_list, start_date, end_date)
        return render_template('result.html', image=image_filename, titles=['Multiple Stocks Close Prices Plot'], choice=choice)
    else:
        return "Invalid choice"


@app.route('/download_csv/<stock_name>/<start_date>/<end_date>')
def download_csv_route(stock_name, start_date, end_date):
    data = yf.download(stock_name, start=start_date, end=end_date)
    output = StringIO()
    data.to_csv(output)
    output.seek(0)
    return send_file(
        BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'{stock_name}_data.csv'
    )

@app.route('/download_pdf/<stock_name>/<start_date>/<end_date>')
def download_pdf_route(stock_name, start_date, end_date):
    data = yf.download(stock_name, start=start_date, end=end_date)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Stock Data for {stock_name}", ln=True, align='C')
    for index, row in data.iterrows():
        pdf.cell(200, 10, txt=str(index.date()) + " " + " ".join(map(str, row.values)), ln=True)
    pdf_output = BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    return send_file(
        pdf_output,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f'{stock_name}_data.pdf'
    )



@app.route('/predict', methods=['POST'])
def predict():
    stock_name = request.form['stock_name']
    start_date = request.form['start_date']
    end_date = request.form['end_date']

    # Modeli oluştur
    input_shape = (60, 1)  # Örnek giriş şekli, kendi modelinizin ihtiyaçlarına göre ayarlayın
    model = create_lstm_model(input_shape)

    # Modeli eğit
    model, scaler = train_lstm_model(stock_name, start_date, end_date, model)
 
    # Tahmin yap
    predictions = predict_future_prices(model, scaler, stock_name, start_date, end_date)

    # Tahmin sonuçlarını render et
    return render_template('result.html', predictions=predictions, stock_name=stock_name)



@app.route('/feedback', methods=['POST'])
def feedback():
    name = request.form['name']
    surname = request.form['surname']
    email = request.form['email']
    feedback = request.form['feedback']
    
    # Geri bildirim verilerini işleme
    with open('feedback.txt', 'a') as file:
        file.write(f"Ad: {name}\nSoyad: {surname}\nE-posta: {email}\nGeri Bildirim: {feedback}\n\n")
    
    return "Geri bildiriminiz başarıyla gönderildi"



@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)




if __name__ == '__main__':
    app.run(port=5001, debug=True)