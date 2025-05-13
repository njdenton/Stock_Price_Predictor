import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import datetime

def get_data(stock_symbol, period="5y"):
    df = yf.download(stock_symbol, period=period)
    df.dropna(inplace=True)
    return df

def add_features(df):
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Return'].rolling(window=10).std()
    df.dropna(inplace=True)
    return df

def train_model(X, y):
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(),
        'SVR': SVR(),
        'KNN': KNeighborsRegressor()
    }
    results = {}
    for name, model in models.items():
        model.fit(X, y)
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        results[name] = (model, mse)
    return results

def forecast_future(df, model, scaler, features, days=5):
    future_dates = []
    last_known = df.copy()

    for _ in range(days):
        row = last_known.iloc[-1:].copy()

        # Simulate next row by using the last values
        next_row = {
            'Open': row['Open'].values[0],
            'High': row['High'].values[0],
            'Low': row['Low'].values[0],
            'Close': row['Close'].values[0],
            'Volume': row['Volume'].values[0],
        }
        # Add rolling features

