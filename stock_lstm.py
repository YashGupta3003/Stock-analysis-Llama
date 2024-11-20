import yfinance as yf
from finvizfinance.quote import finvizfinance
import pandas as pd
import numpy as np
import holidays
import streamlit as st
import requests
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

#############
##functions##
#############

# Function to fetch news data
def get_news_data(ticker):
    try:
        stock = finvizfinance(ticker)
        news_df = stock.ticker_news()
        news_df['Title'] = news_df['Title'].str.lower()
        news_df['Sentiment'] = news_df['Title'].apply(lambda x: 'positive' if 'good' in x else 'negative')
        news_df['Date'] = pd.to_datetime(news_df['Date'])
        news_df['DateOnly'] = news_df['Date'].dt.date
        return news_df
    except Exception as e:
        st.error(f"Error fetching data for ticker {ticker}: {e}")
        return pd.DataFrame()

# Function to process sentiment data
def process_sentiment_data(news_df):
    grouped = news_df.groupby(['DateOnly', 'Sentiment']).size().unstack(fill_value=0)
    grouped = grouped.reindex(columns=['positive', 'negative'], fill_value=0)
    grouped['7day_avg_positive'] = grouped['positive'].rolling(window=7, min_periods=1).sum()
    grouped['7day_avg_negative'] = grouped['negative'].rolling(window=7, min_periods=1).sum()
    grouped['7day_pct_positive'] = grouped['positive'] / (grouped['positive'] + grouped['negative'])
    result_df = grouped.reset_index()
    return result_df

# Function to get stock data
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data['Pct_Change'] = stock_data['Close'].pct_change() * 100
    return stock_data

# Function to combine data
def combine_data(result_df, stock_data):
    combined_df = result_df.set_index('DateOnly').join(stock_data['Pct_Change'], how='inner')
    combined_df['Lagged_7day_pct_positive'] = combined_df['7day_pct_positive'].shift(1)
    combined_df.dropna(inplace=True)
    return combined_df

# Function to prepare data for LSTM
def prepare_data_for_lstm(combined_df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(combined_df[['Pct_Change', 'Lagged_7day_pct_positive']])
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, :])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    return X, y, scaler

# Function to create LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to create plot
def create_plot(combined_df, y_test, predictions, scaler):
    y_test = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((len(y_test), 1))), axis=1))[:, 0]
    predictions = scaler.inverse_transform(np.concatenate((predictions.reshape(-1, 1), np.zeros((len(predictions), 1))), axis=1))[:, 0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=combined_df.index[-len(y_test):],
        y=y_test,
        name='Actual Price Change',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=combined_df.index[-len(predictions):],
        y=predictions,
        name='Predicted Price Change',
        line=dict(color='red')
    ))
    st.plotly_chart(fig)

############
# STREAMLIT
############

st.sidebar.title("Predicting Stock Prices with LSTM")
ticker = st.sidebar.text_input("Enter stock ticker (e.g., SBUX):", value='SBUX')
run_button = st.sidebar.button("Run Analysis")

if run_button:
    news_df = get_news_data(ticker)
    result_df = process_sentiment_data(news_df)
    start_date = result_df['DateOnly'].min().strftime('%Y-%m-%d')
    end_date = result_df['DateOnly'].max().strftime('%Y-%m-%d')
    stock_data = get_stock_data(ticker, start_date, end_date)
    combined_df = combine_data(result_df, stock_data)
    
    X, y, scaler = prepare_data_for_lstm(combined_df)
    model = create_lstm_model(X.shape[1:])
    model.fit(X, y, epochs=10, batch_size=32)

    X_test = X[-len(y):]
    y_test = y[-len(y):]
    predictions = model.predict(X_test)
    create_plot(combined_df, y_test, predictions, scaler)
