import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def get_data(stock_symbol, period='5y'):
    """
    Download historical data for a stock symbol from Yahoo Finance.
    """
    df = yf.download(stock_symbol, period=period)
    df.dropna(inplace=True)
    return df

#
def add_features(df):
    """
    Add additional features (technical indicators) to the DataFrame for model training.
    """
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Return'].rolling(window=10).std()
    df.dropna(inplace=True)
    return df

def train_model(X, y):
    """
    Train multiple machine learning models and evaluate their mean squared error.
    Returns the models and their MSEs.
    """
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
    """
    Forecast future stock prices using the trained model.
    It uses the last known row and extends it iteratively day-by-day.
    """
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
        window = last_known.tail(50)    # Enough for SMA_50
        next_row['SMA_10'] = window['Close'].rolling(10).mean().iloc[-1]
        next_row['SMA_50'] = window['Close'].rolling(50).mean().iloc[-1]
        next_row['Return'] = window['Close'].pct_change().iloc[-1]
        next_row['Volatility'] = window['Return'].rolling(10).std().iloc[-1]

        # Convert to dataframe and drop missing values
        next_features = pd.DataFrame([next_row])[features]

        if next_features[features].isnull().values.any():
            print('Skipping day due to missing features')
            break

        # Scale and predict
        next_scaled = scaler.transform(next_features)
        next_close = model.predict(next_scaled)[0]

        # Add next valid date
        new_date = last_known.index[-1] + pd.Timedelta(days=1)
        while new_date.weekday() >= 5:  # Skip weekends
            new_date += pd.Timedelta(days=1)

        future_dates.append((new_date, next_close))

        # Append synthetic row for next iteration
        new_row = row.copy()
        new_row.index = [new_date]
        new_row['Close'] = next_close
        last_known = pd.concat([last_known, new_row])

    return future_dates

def run_prediction(stock_symbol, forecast_days):
    """
    End-to-end function to:
    - Fetch data
    - Engineer features
    - Train models
    - Select the best model
    - Forecast future prices
    """
    df = get_data(stock_symbol)
    df = add_features(df)

    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'SMA_50', 'Return', 'Volatility']
    df['Prediction'] = df['Close'].shift(-forecast_days)
    df.dropna(inplace=True)

    x = df[features].values
    y = df['Prediction'].values

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    model_results = train_model(x_scaled, y)
    best_model_name = min(model_results, key=lambda x: model_results[x][1])
    best_model, best_mse = model_results[best_model_name]

    future_forecast = forecast_future(df, best_model, scaler, features, forecast_days)
    return {
        'historical': df.copy(),
        'forecast': future_forecast,
        'model': best_model_name,
        'mse': best_mse
    }
