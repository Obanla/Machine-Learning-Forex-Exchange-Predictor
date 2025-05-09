import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import tensorflow as tf
import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import load_model

# --- Configuration ---
MODELS_DIR = "LSTM_Models"
SCALERS_DIR = "LSTM_Scalers"

# Mapping between display names and cleaned base names
CURRENCY_MAPPING = {
    "AUSTRALIA - AUSTRALIAN DOLLAR/US$": "AUSTRALIA_-_AUSTRALIAN_DOLLAR_US$",
    "BRAZIL - REAL/US$": "BRAZIL_-_REAL_US$",
    "CANADA - CANADIAN DOLLAR/US$": "CANADA_-_CANADIAN_DOLLAR_US$",
    "CHINA - YUAN/US$": "CHINA_-_YUAN_US$",
    "DENMARK - DANISH KRONE/US$": "DENMARK_-_DANISH_KRONE_US$",
    "EURO AREA - EURO/US$": "EURO_AREA_-_EURO_US$",
    "HONG KONG - HONG KONG DOLLAR/US$": "HONG_KONG_-_HONG_KONG_DOLLAR_US$",
    "INDIA - INDIAN RUPEE/US$": "INDIA_-_INDIAN_RUPEE_US$",
    "JAPAN - YEN/US$": "JAPAN_-_YEN_US$",
    "KOREA - WON/US$": "KOREA_-_WON_US$",
    "MALAYSIA - RINGGIT/US$": "MALAYSIA_-_RINGGIT_US$",
    "MEXICO - MEXICAN PESO/US$": "MEXICO_-_MEXICAN_PESO_US$",
    "NEW ZEALAND - NEW ZELAND DOLLAR/US$": "NEW_ZEALAND_-_NEW_ZELAND_DOLLAR_US$",
    "NORWAY - NORWEGIAN KRONE/US$": "NORWAY_-_NORWEGIAN_KRONE_US$",
    "SINGAPORE - SINGAPORE DOLLAR/US$": "SINGAPORE_-_SINGAPORE_DOLLAR_US$",
    "SOUTH AFRICA - RAND/US$": "SOUTH_AFRICA_-_RAND_US$",
    "SRI LANKA - SRI LANKAN RUPEE/US$": "SRI_LANKA_-_SRI_LANKAN_RUPEE_US$",
    "SWEDEN - KRONA/US$": "SWEDEN_-_KRONA_US$",
    "SWITZERLAND - FRANC/US$": "SWITZERLAND_-_FRANC_US$",
    "TAIWAN - NEW TAIWAN DOLLAR/US$": "TAIWAN_-_NEW_TAIWAN_DOLLAR_US$",
    "THAILAND - BAHT/US$": "THAILAND_-_BAHT_US$",
    "UNITED KINGDOM - UNITED KINGDOM POUND/US$": "UNITED_KINGDOM_-_UNITED_KINGDOM_POUND_US$",
}


def create_uni_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0  # Avoid division by zero
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100


def main():
    st.set_page_config(page_title="Forex LSTM Predictor", page_icon="📈", layout="wide")
    st.title("💹 Forex Exchange Rate Predictor with LSTM")

    # Load models and scalers
    models, scalers = {}, {}

    # Load models
    if os.path.exists(MODELS_DIR):
        for f in os.listdir(MODELS_DIR):
            if f.endswith(".keras"):
                base_name = f.replace("_best_model.keras", "").replace(".keras", "")
                models[base_name] = load_model(os.path.join(MODELS_DIR, f))

    # Load scalers
    if os.path.exists(SCALERS_DIR):
        for f in os.listdir(SCALERS_DIR):
            if f.endswith(".pkl"):
                base_name = f.replace("scaler_", "").replace(".pkl", "")
                scalers[base_name] = joblib.load(os.path.join(SCALERS_DIR, f))

    # Load historical data
    try:
        df = pd.read_csv('df_Prophet_Autots.csv',
                         parse_dates=['Time Serie'],
                         index_col='Time Serie')
    except FileNotFoundError:
        st.error("Historical data file not found")
        return

    # UI Components
    currency = st.sidebar.selectbox("Select Currency", df.columns)
    test_size = st.sidebar.slider("Test Set Size (days)", 20, 120, 60)
    input_sequence_length = 30

    # Get base name from mapping
    base_name = CURRENCY_MAPPING.get(currency)
    if not base_name:
        st.error(f"No configuration found for {currency}")
        return

    # Get model and scaler
    model = models.get(base_name)
    scaler = scalers.get(base_name)

    if not model:
        st.error(f"No LSTM model available for {currency}")
        return
    if not scaler:
        st.error(f"No scaler found for {currency}")
        return

    st.header(f"Actual vs Predicted: {currency}")

    # Plot historical data
    fig, ax = plt.subplots(figsize=(12, 6))
    df[currency].plot(ax=ax, title="Historical Data")
    st.pyplot(fig)


    if st.button("Generate Test Set Forecast"):
        with st.spinner("Generating predictions..."):
            try:
                series = df[currency].dropna()
                train, test = series.iloc[:-test_size], series.iloc[-test_size:]

                # Scale data
                train_scaled = scaler.transform(train.values.reshape(-1, 1))
                test_scaled = scaler.transform(test.values.reshape(-1, 1))

                # Walk-forward prediction
                history = list(train_scaled.flatten())
                predictions_scaled = []

                for i in range(len(test_scaled)):
                    input_seq = np.array(history[-input_sequence_length:]).reshape(1, input_sequence_length, 1)
                    yhat = model.predict(input_seq, verbose=0)[0][0]
                    predictions_scaled.append(yhat)
                    history.append(test_scaled[i][0])

                # Inverse transformations
                preds_denorm = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()
                test_denorm = scaler.inverse_transform(test_scaled).flatten()

                # Plot results
                fig, ax = plt.subplots(figsize=(14, 6))
                ax.plot(test.index, test_denorm, label='Actual', marker='o', color='blue')
                ax.plot(test.index, preds_denorm, label='Predicted', linestyle='--', marker='x', color='orange')
                ax.set_title(f"{currency} Forecast")
                ax.legend()
                st.pyplot(fig)


                # Metrics
                current = test_denorm[-1]
                predicted = preds_denorm[-1]

                rmse = np.sqrt(mean_squared_error(test_denorm, preds_denorm))
                mae = mean_absolute_error(test_denorm, preds_denorm)
                mape = calculate_mape(test_denorm, preds_denorm)

                # Display metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Last Actual Rate", f"{current:.4f}")
                col2.metric("Last Predicted Rate", f"{predicted:.4f}", delta=f"{predicted - current:.4f}")
                col3.metric("RMSE", f"{rmse:.4f}")
                col4.metric("MAE", f"{mae:.4f}")

                # Display MAPE in its own row
                st.metric("MAPE", f"{mape:.2f}%")

            except Exception as e:
                st.error(f"Forecasting failed: {str(e)}")


if __name__ == "__main__":
    main()