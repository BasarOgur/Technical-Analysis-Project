import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

stock_name = input("Lütfen girmek istediğiniz hisse senedi adını giriniz: ")

data = yf.download(stock_name, start="2020-01-01", end="2021-01-01")
# print(data.head()) # Fetching Historical Data

apple = yf.Ticker(stock_name)
# print(apple.info)  # General information about Apple Inc.

recent_data = yf.download(stock_name, period="5d")
# print(recent_data) # Getting Recent Data

multi_data = yf.download([stock_name, "MSFT"], start="2020-01-01", end="2021-01-01")
#print(multi_data) # Retrieve data for multiple stocks in one go

data = yf.download(stock_name, start="2020-01-01", end="2021-01-01", auto_adjust=True)
# print(data['Close'])  # This will show the adjusted close prices

weekly_data = yf.download(stock_name, start="2020-01-01", end="2021-01-01", interval="1wk")
# print(weekly_data) 

apple = yf.Ticker(stock_name)
dividends = apple.dividends
splits = apple.splits
# print(dividends, splits) # Access dividend and stock split history.

apple = yf.Ticker(stock_name)
#print(apple.history(period="1d")) # Fetch the most up-to-date stock price.

data = yf.download(stock_name, start="2022-01-01", end="2022-12-31")
#print(data)



data = yf.download(stock_name, start="2020-01-01", end="2024-08-05")
data['Close'].plot()
plt.title(f"{stock_name} Stock Prices")
# plt.show()






# 2.sayfadaki bilgiler, üstekilerden biraz farklı 
# Set the start and end date
start_date = '1990-01-01'
end_date = '2021-07-12'
# Set the ticker
ticker = 'AMZN'
# Get the data
data = yf.download(ticker, start_date, end_date)
# Print 5 rows
# print(data.tail())




# Set the start and end date
start_date = '1990-01-01'
end_date = '2021-07-12'

# Define the ticker list
tickers_list = ['AAPL', 'IBM', 'MSFT', 'WMT']

# Create placeholder for data
data = pd.DataFrame(columns=tickers_list)

# Fetch the data
for ticker in tickers_list:
    data[ticker] = yf.download(ticker, 
                               start_date,
                               end_date)['Adj Close']
    
# Print first 5 rows of the data
# print(data.head())

# Plot all the close prices
data.plot(figsize=(10, 7))

# Show the legend
plt.legend()

# Define the label for the title of the figure
plt.title("Adjusted Close Price", fontsize=16)

# Define the labels for x-axis and y-axis
plt.ylabel('Price', fontsize=14)
plt.xlabel('Year', fontsize=14)

# Plot the grid lines
plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
plt.show()