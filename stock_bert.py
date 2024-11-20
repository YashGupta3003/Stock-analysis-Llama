import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import holidays
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from newsapi import NewsApiClient

#############
## Functions ##
#############

# Load pre-trained FinBERT model and tokenizer
@st.cache_resource()
def load_finbert_model():
    tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return nlp

nlp = load_finbert_model()

# Function for sentiment classification using FinBERT
def classify_sentiment_finbert(title):
    result = nlp(title)
    sentiment = result[0]['label']
    return sentiment.upper()  # Convert to uppercase for consistency

# Function to get news data using NewsAPI
def get_news_data_newsapi(ticker, api_key):
    try:
        newsapi = NewsApiClient(api_key='732fa4fe2c4a4c56a24041307654200e')
        all_articles = newsapi.get_everything(q=ticker, language='en', sort_by='relevancy', page_size=100)
        news_df = pd.DataFrame(all_articles['articles'])
        news_df['Title'] = news_df['title'].str.lower()
        news_df['Date'] = pd.to_datetime(news_df['publishedAt'])
        news_df['DateOnly'] = news_df['Date'].dt.date

        # Classify sentiment using FinBERT
        news_df['Sentiment'] = news_df['Title'].apply(classify_sentiment_finbert)

        # Filter out neutral sentiment
        news_df = news_df[news_df['Sentiment'] != 'NEUTRAL']

        return news_df

    except Exception as e:
        st.error(f"Error fetching data for ticker {ticker}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if there's an error

# Function to group and process sentiment data
def process_sentiment_data(news_df):
    grouped = news_df.groupby(['DateOnly', 'Sentiment']).size().unstack(fill_value=0)
    grouped = grouped.reindex(columns=['POSITIVE', 'NEGATIVE'], fill_value=0)
    grouped['7day_avg_positive'] = grouped['POSITIVE'].rolling(window=7, min_periods=1).sum()
    grouped['7day_avg_negative'] = grouped['NEGATIVE'].rolling(window=7, min_periods=1).sum()
    grouped['7day_pct_positive'] = grouped['POSITIVE'] / (grouped['POSITIVE'] + grouped['NEGATIVE'])
    result_df = grouped.reset_index()
    return result_df

# Function to fetch and process stock data
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data['Pct_Change'] = stock_data['Close'].pct_change() * 100
    return stock_data

# Function to combine sentiment and stock data
def combine_data(result_df, stock_data):
    combined_df = result_df.set_index('DateOnly').join(stock_data['Pct_Change'], how='inner')
    combined_df['Lagged_7day_pct_positive'] = combined_df['7day_pct_positive'].shift(1)
    return combined_df

# Function to calculate Pearson correlation
def calculate_correlation(combined_df):
    correlation_pct_change = combined_df[['Lagged_7day_pct_positive', 'Pct_Change']].corr().iloc[0, 1]
    return correlation_pct_change

# Function to get future dates excluding weekends and holidays
def get_future_dates(start_date, num_days):
    us_holidays = holidays.India()
    future_dates = []
    current_date = start_date
    while len(future_dates) < num_days:
        if current_date.weekday() < 5 and current_date not in us_holidays:
            future_dates.append(current_date)
        current_date += pd.Timedelta(days=1)
    return future_dates

# Function to fit ARIMAX model and forecast
def fit_and_forecast(combined_df, forecast_steps=3):
    endog = combined_df['Pct_Change'].dropna()
    exog = combined_df['Lagged_7day_pct_positive'].dropna()
    endog, exog = endog.loc[exog.index], exog
    
    model = SARIMAX(endog, exog=exog, order=(1, 1, 1))
    fit = model.fit(disp=False)
    
    future_dates = get_future_dates(combined_df.index[-1], forecast_steps)
    future_exog = combined_df['Lagged_7day_pct_positive'][-forecast_steps:].values.reshape(-1, 1)
    
    forecast = fit.get_forecast(steps=forecast_steps, exog=future_exog)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()
    
    return forecast_mean, forecast_ci, future_dates

# Function to create and display plot
def create_plot(combined_df, forecast_mean, forecast_ci, forecast_index):
    sentiment_std = (combined_df['7day_pct_positive'] - combined_df['7day_pct_positive'].mean()) / combined_df['7day_pct_positive'].std()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=combined_df.index,
        y=sentiment_std,
        name='Standardized Sentiment Proportion',
        line=dict(color='blue'),
        mode='lines'
    ))

    fig.add_trace(go.Scatter(
        x=combined_df.index,
        y=combined_df['Pct_Change'],
        name='Stock Pct Change',
        line=dict(color='green'),
        yaxis='y2',
        mode='lines'
    ))

    fig.add_trace(go.Scatter(
        x=forecast_index,
        y=forecast_mean,
        name='Forecasted Pct Change',
        line=dict(color='red'),
        mode='lines'
    ))

    fig.add_trace(go.Scatter(
        x=np.concatenate([forecast_index, forecast_index[::-1]]),
        y=np.concatenate([forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1][::-1]]),
        fill='toself',
        fillcolor='rgba(255,0,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ))

    fig.update_layout(
        title='Sentiment Proportion and Stock Percentage Change with Forecast',
        xaxis_title='Date',
        yaxis=dict(
            title='Standardized Sentiment Proportion',
            titlefont=dict(color='blue')
        ),
        yaxis2=dict(
            title='Stock Pct Change',
            titlefont=dict(color='green'),
            overlaying='y',
            side='right'
        ),
        template='plotly_dark'
    )

    st.plotly_chart(fig)

############
### Streamlit App ###
############

# Streamlit app
st.sidebar.title("Predicting Stock Prices by News Sentiment")
ticker = st.sidebar.text_input("Enter stock ticker (e.g., SBUX for US stocks or RELIANCE.NS for Indian stocks):", value='RELIANCE.NS')
api_key = st.sidebar.text_input("Enter NewsAPI key:", type="password")
run_button = st.sidebar.button("Run Analysis")

if run_button:
    news_df = get_news_data_newsapi(ticker, api_key)
    result_df = process_sentiment_data(news_df)
    start_date = result_df['DateOnly'].min().strftime('%Y-%m-%d')
    end_date = result_df['DateOnly'].max().strftime('%Y-%m-%d')
    stock_data = get_stock_data(ticker, start_date, end_date)
    combined_df = combine_data(result_df, stock_data)
    correlation_pct_change = calculate_correlation(combined_df)
    st.write(f'Pearson correlation between lagged sentiment score and stock percentage change: {correlation_pct_change}')
    forecast_mean, forecast_ci, forecast_index = fit_and_forecast(combined_df)
    create_plot(combined_df, forecast_mean, forecast_ci, forecast_index)
