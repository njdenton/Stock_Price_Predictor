import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from predictor import run_prediction

# Set up Streamlit page
st.set_page_config(page_title='Stock Price Predictor', page_icon=':chart_with_upwards_trend:', layout='centered')
st.title('Stock Price Forecasting App')

# Inputs
stock_symbol = st.text_input('Enter Stock Symbol: ', 'AAPL')
forecast_days = st.slider('Select number of days to forecast:', 1, 30, 5)
lookback_days = st.slider('Days of historical data to show on graph:', 30, 180, 90)

# When user clicks predict
if st.button('Predict'):
    with st.spinner('Fetching data and training models...'):
        result = run_prediction(stock_symbol, forecast_days)

    # Show model info
    st.success(f'Best model: {result['model']} (MSE: {result['mse']:.4f})')

    # Get historical and forecast data
    df = result['historical']
    future_df = pd.DataFrame(result['forecast'], columns=['Date', 'Forecast'])
    future_df['Date'] = pd.to_datetime(future_df['Date'], errors='coerce')
    future_df['Forecast'] = pd.to_numeric(future_df['Forecast'], errors='coerce')
    future_df.dropna(inplace=True)
    future_df.sort_values('Date', inplace=True)
    future_df.set_index('Date', inplace=True)
    if future_df.empty:
        st.error('No forecast data available.')
        st.stop()

    # Prepare recent historical data
    recent_df = df.tail(lookback_days).copy()
    last_date = recent_df.index[-1]
    last_close = recent_df['Close'].iloc[-1]

    # Create forecast line that connects to last actual close
    combined_forecast = pd.concat([
        pd.DataFrame({'Forecast': [last_close]}, index=[last_date]),
        future_df
    ])
    combined_forecast.sort_index(inplace=True)

    # Ensure the index is datetime and Forecast is numeric
    combined_forecast.index = pd.to_datetime(combined_forecast.index, errors='coerce')
    combined_forecast['Forecast'] = pd.to_numeric(combined_forecast['Forecast'], errors='coerce')
    combined_forecast.dropna(subset=['Forecast'], inplace=True)

    # Check again to avoid breaking
    if combined_forecast.empty:
        st.error('Forecast data could not be plotted due to missing or invalid values.')
        st.stop()

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    recent_df['Close'].plot(ax=ax, label='Historical Close', color='blue')
    combined_forecast['Forecast'].plot(ax=ax, label='Forecast', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'{stock_symbol} - Forecast for Next {forecast_days} Days')
    plt.legend()
    plt.grid(True)
    st.pyplot(fig)

    # Forecast table
    st.subheader('Forecast Data')
    st.dataframe(future_df)

    # Optional download as CSV
    st.download_button(
        label='Download forecast data as CSV file',
        data=future_df.to_csv().encode('utf-8'),
        file_name=f'{stock_symbol}_forecast_{forecast_days}_days.csv',
        mime='text/csv'
    )
