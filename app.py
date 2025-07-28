import streamlit as st # For creating the web app UI
import yfinance as yf # For fetching live stock data
import pandas as pd # For data manipulation
import numpy as np # For numerical operations
import pandas_ta as ta # For calculating technical indicators
from tensorflow.keras.models import load_model # For loading your trained model
import joblib # For loading your scaler
from datetime import datetime, timedelta # For date calculations
import matplotlib.pyplot as plt # For plotting

# --- Configuration (Must match your training setup) ---
# These values must be exactly the same as what you used when training your model in Colab.
LOOK_BACK_DAYS = 60 # Number of previous days to use for prediction (sequence length for LSTM)
FEATURES_TO_USE = [
    'Close', 'Volume', 'SMA_10', 'SMA_20', 'EMA_12', 'EMA_26', 'RSI',
    'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
    'BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0'
]
NUM_FEATURES = len(FEATURES_TO_USE)

# --- Load Model and Scaler ---
# These files should be in a subfolder named 'model_artifacts' in your GitHub repo.
# When deploying to Streamlit Cloud, the path will be relative to your app.py file.
MODEL_PATH = 'model_artifacts/my_lstm_stock_predictor.keras'
SCALER_PATH = 'model_artifacts/minmax_scaler_all_features.pkl'

# Use Streamlit's caching mechanism to load the model only once
@st.cache_resource # This is the new decorator for caching models/large objects
def load_my_model():
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model from {MODEL_PATH}: {e}")
        st.stop()

@st.cache_resource
def load_my_scaler():
    try:
        scaler = joblib.load(SCALER_PATH)
        return scaler
    except Exception as e:
        st.error(f"Error loading scaler from {SCALER_PATH}: {e}")
        st.stop()

model = load_my_model()
scaler = load_my_scaler()

# --- Helper Function for Data Preprocessing (similar to your Colab notebook) ---
def preprocess_data_for_prediction(df_input, ticker_symbol):
    # 1. Flatten MultiIndex Columns (if yfinance downloads fresh data)
    if isinstance(df_input.columns, pd.MultiIndex):
        df_input.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df_input.columns.values]
    df_input.rename(columns={
        f'Close_{ticker_symbol}': 'Close', f'High_{ticker_symbol}': 'High', f'Low_{ticker_symbol}': 'Low',
        f'Open_{ticker_symbol}': 'Open', f'Volume_{ticker_symbol}': 'Volume'
    }, inplace=True)

    df_proc = df_input.copy()

    # 2. Add Technical Indicators (ensure this matches your training)
    df_proc['SMA_10'] = ta.sma(df_proc['Close'], length=10)
    df_proc['SMA_20'] = ta.sma(df_proc['Close'], length=20)
    df_proc['EMA_12'] = ta.ema(df_proc['Close'], length=12)
    df_proc['EMA_26'] = ta.ema(df_proc['Close'], length=26)
    df_proc['RSI'] = ta.rsi(df_proc['Close'], length=14)

    macd = ta.macd(df_proc['Close'])
    if macd is not None and not macd.empty:
        df_proc = df_proc.join(macd)

    bbands = ta.bbands(df_proc['Close'])
    if bbands is not None and not bbands.empty:
        df_proc = df_proc.join(bbands)

    df_proc.dropna(inplace=True)

    # Check if enough data for prediction after dropping NaNs
    if len(df_proc) < LOOK_BACK_DAYS:
        raise ValueError(f"Not enough historical data ({len(df_proc)} rows) to create a sequence of {LOOK_BACK_DAYS} days after calculating indicators. Try a longer start date or a stock with more data.")

    # Select and scale the last 'LOOK_BACK_DAYS' worth of data using the *trained* scaler
    last_n_days_data = df_proc[FEATURES_TO_USE].tail(LOOK_BACK_DAYS).values
    scaled_last_n_days = scaler.transform(last_n_days_data)

    # Reshape for LSTM: (1, LOOK_BACK_DAYS, NUM_FEATURES)
    return scaled_last_n_days.reshape(1, LOOK_BACK_DAYS, NUM_FEATURES), df_proc.index[-1] # Also return the last date

# --- Streamlit UI ---
st.title('Stock Price Prediction & Forecasting with LSTM')
st.write('Predict the next trading day\'s closing price for any stock using an LSTM model!')

ticker_input = st.text_input('Enter Stock Ticker (e.g., AAPL, GOOGL, RELIANCE.NS):', 'AAPL').upper()

if st.button('Predict Next Day Close'):
    if not ticker_input:
        st.warning("Please enter a stock ticker.")
    else:
        with st.spinner(f'Fetching data and predicting for {ticker_input}...'):
            try:
                # Fetch enough data for look-back + longest indicator period (e.g., MACD is 26)
                # Add some buffer (e.g., 200 days) to ensure all indicators can be calculated
                end_date = datetime.now()
                start_date = end_date - timedelta(days=LOOK_BACK_DAYS + 100) # Ensure enough data for indicators

                df_live = yf.download(ticker_input, start=start_date, end=end_date)

                if df_live.empty:
                    st.error(f"Could not fetch data for {ticker_input}. Please check the ticker symbol or try again later.")
                    st.stop()

                # Preprocess the live data
                processed_input, last_actual_date = preprocess_data_for_prediction(df_live, ticker_input)

                # Make prediction
                scaled_prediction = model.predict(processed_input)

                # Inverse transform the prediction
                # Create a dummy array with the correct number of features for inverse_transform
                dummy_prediction_array = np.zeros((1, NUM_FEATURES))
                dummy_prediction_array[:, 0] = scaled_prediction[0, 0] # Put prediction into the 'Close' price slot
                predicted_price = scaler.inverse_transform(dummy_prediction_array)[:, 0][0]

                st.success(f"**Predicted next trading day's closing price for {ticker_input}: ${predicted_price:.2f}**")

                # --- Visualization ---
                st.subheader(f'Recent {ticker_input} Prices & Prediction')

                # Ensure enough recent data for plotting
                plot_data = df_live['Close'].tail(LOOK_BACK_DAYS * 2) 

                # Get the date of the predicted price (next trading day)
                next_trading_day = last_actual_date + pd.Timedelta(days=1)
                while next_trading_day.dayofweek > 4: # Skip Saturday (5) and Sunday (6)
                    next_trading_day += pd.Timedelta(days=1)

                # Create a DataFrame for plotting, including the prediction
                plot_df = pd.DataFrame({
                    'Price': plot_data.values
                }, index=plot_data.index)

                # Add prediction as a new row to the plot_df
                plot_df.loc[next_trading_day] = predicted_price

                st.line_chart(plot_df)
                st.write(f"The blue line shows the actual closing prices. The last point (blue) is for {last_actual_date.strftime('%Y-%m-%d')}. The green dot is the predicted price for the next trading day ({next_trading_day.strftime('%Y-%m-%d')}).")

            except ValueError as ve:
                st.error(f"Prediction Error: {ve}")
            except Exception as e:
                st.error(f"An unexpected error occurred during prediction: {e}")

st.markdown("---")
st.markdown("This application demonstrates an LSTM model's ability to forecast stock prices based on historical data and technical indicators. Remember, stock price prediction is inherently complex and carries high risk; this app is for educational purposes only and not financial advice.")
