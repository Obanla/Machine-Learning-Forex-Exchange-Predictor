import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import load_model
from statsmodels.tsa.arima.model import ARIMAResults
from autots import AutoTS


# Model Configuration
ARIMA_MODELS_DIR = "Arima_Models"
ARIMA_SCALERS_DIR = "Arima_Scalers"
LSTM_MODELS_DIR = "LSTM_Models"
LSTM_SCALERS_DIR = "LSTM_Scalers"
AUTOTS_MODEL_PATH = "autots_model.joblib"

# Mapping between display names and cleaned base names for the different currencies
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

# Preparing the data for the LSTM model
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

#To calculate the MAPE metric
def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

# Function to load the different Models (Arima, LSTM and Autots)
def load_models(model_type):
    models, scalers = {}, {}

    if model_type == "AutoTS":
        try:
            models['autots'] = joblib.load(AUTOTS_MODEL_PATH)
        except FileNotFoundError:
            st.error("AutoTS model file not found")
        return models, scalers

    model_dir = ARIMA_MODELS_DIR if model_type == "ARIMA" else LSTM_MODELS_DIR
    scaler_dir = ARIMA_SCALERS_DIR if model_type == "ARIMA" else LSTM_SCALERS_DIR

    if os.path.exists(model_dir):
        for f in os.listdir(model_dir):
            if model_type == "ARIMA" and f.startswith("model_"):
                base_name = f.replace("model_", "").replace(".joblib", "")
                models[base_name] = joblib.load(os.path.join(model_dir, f))
            elif model_type == "LSTM" and f.endswith(".keras"):
                base_name = f.replace(".keras", "")
                models[base_name] = load_model(os.path.join(model_dir, f))

    if os.path.exists(scaler_dir):
        for f in os.listdir(scaler_dir):
            if f.endswith((".joblib", ".pkl")):
                base_name = f.replace("scaler_", "").replace(".joblib", "").replace(".pkl", "")
                scalers[base_name] = joblib.load(os.path.join(scaler_dir, f))

    return models, scalers

# Function controlling the generation of forecasts
def run_forecast(model_type, base_name, test_size, series, models, scalers):
    test = series.iloc[-test_size:]
    train = series.iloc[:-test_size]
    input_sequence_length = 30 if model_type == "LSTM" else None

#Generating forecasts for the Autots model
    if model_type == "AutoTS":
        model = models.get('autots')
        if model is None:
            raise ValueError("AutoTS model not loaded")
        model.forecast_length = test_size
        model.fit_data(train.to_frame())
        prediction = model.predict()
        forecast = prediction.forecast.set_index(test.index)
        upper = prediction.upper_forecast.set_index(test.index)
        lower = prediction.lower_forecast.set_index(test.index)
        preds_denorm = forecast[series.name].values
        test_denorm = test.values

    # For Arima and LSTM since each currency pair has its own model
    else:
        model = models.get(base_name)
        scaler = scalers.get(base_name)
        if model is None or scaler is None:
            raise ValueError(f"{model_type} model or scaler not found for {base_name}")

        train_scaled = scaler.transform(train.values.reshape(-1, 1)).flatten()
        test_scaled = scaler.transform(test.values.reshape(-1, 1)).flatten()
        predictions_scaled = []
        history = list(train_scaled)

        #Generating forecasts for the Arima Model

        if model_type == "ARIMA":
            for t in range(len(test_scaled)):
                fitted = model.fit(history)
                yhat = fitted.predict()[0]
                predictions_scaled.append(yhat)
                history.append(test_scaled[t])

        #forecasts for the lstm model
        else:
            for i in range(len(test_scaled)):
                input_seq = np.array(history[-input_sequence_length:]).reshape(1, input_sequence_length, 1)
                yhat = model.predict(input_seq, verbose=0)[0][0]
                predictions_scaled.append(yhat)
                history.append(test_scaled[i])

        #Inversing the applied normalization so that results are returned in the original currency values

        preds_denorm = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()
        test_denorm = scaler.inverse_transform(np.array(test_scaled).reshape(-1, 1)).flatten()
        upper, lower = None, None  # Not applicable

    # Calculating RMSE, MAE and MAPE metrics
    rmse = np.sqrt(mean_squared_error(test_denorm, preds_denorm))
    mae = mean_absolute_error(test_denorm, preds_denorm)
    mape = calculate_mape(test_denorm, preds_denorm)

    return test.index, test_denorm, preds_denorm, rmse, mae, mape, upper, lower

# Main display page
def main():
    st.set_page_config(page_title="Forex Predictor", page_icon="📈", layout="wide")
    st.title("💹 Forex Exchange Rate Predictor")

    # Loading the historical data
    try:
        df = pd.read_csv('df_Prophet_Autots.csv',
                         parse_dates=['Time Serie'],
                         index_col='Time Serie')
    except FileNotFoundError:
        st.error("Historical data file not found")
        return

    # Setting the UI Components
    currency = st.sidebar.selectbox("Select Currency", df.columns)
    model_type = st.sidebar.radio("Select Model Type", ["ARIMA", "LSTM", "AutoTS"])
    test_size = st.sidebar.slider("Test Set Size (days)", 20, 120, 60)

    # Adding a button for model comparison
    compare_models = st.sidebar.button("🔀 Compare All Models")

    # Get base name for each currency from the mapping
    base_name = CURRENCY_MAPPING.get(currency)
    if not base_name:
        st.error(f"No configuration found for {currency}")
        return

    # Load all models and scalers once
    arima_models, arima_scalers = load_models("ARIMA")
    lstm_models, lstm_scalers = load_models("LSTM")
    autots_models, _ = load_models("AutoTS")

    # Show historical data
    st.header(f"Historical Exchange Rates: {currency}")
    fig, ax = plt.subplots(figsize=(12, 6))
    df[currency].plot(ax=ax, title="Historical Data")
    st.pyplot(fig)

    if compare_models:
        st.header(f"Model Comparison for {currency}")
        series = df[currency].dropna()
        metrics_dict = {}

        # ARIMA
        try:
            _, _, _, rmse, mae, mape, _, _ = run_forecast("ARIMA", base_name, test_size, series, arima_models, arima_scalers)
            metrics_dict["ARIMA"] = {"RMSE": rmse, "MAE": mae, "MAPE": mape}
        except Exception as e:
            st.warning(f"ARIMA model failed: {e}")

        # LSTM
        try:
            _, _, _, rmse, mae, mape, _, _ = run_forecast("LSTM", base_name, test_size, series, lstm_models, lstm_scalers)
            metrics_dict["LSTM"] = {"RMSE": rmse, "MAE": mae, "MAPE": mape}
        except Exception as e:
            st.warning(f"LSTM model failed: {e}")

        # AutoTS
        try:
            _, _, _, rmse, mae, mape, _, _ = run_forecast("AutoTS", base_name, test_size, series, autots_models, {})
            metrics_dict["AutoTS"] = {"RMSE": rmse, "MAE": mae, "MAPE": mape}
        except Exception as e:
            st.warning(f"AutoTS model failed: {e}")

        if metrics_dict:
            comp_df = pd.DataFrame(metrics_dict).T
            st.dataframe(comp_df.style.format({
                "RMSE": "{:.4f}",
                "MAE": "{:.4f}",
                "MAPE": "{:.2f}%"
            }), use_container_width=True)

            # Plot bar charts for metrics
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            for i, metric in enumerate(["MAE", "RMSE", "MAPE"]):
                comp_df[metric].plot(kind='bar', ax=axes[i], title=metric)
                axes[i].set_xticklabels(comp_df.index, rotation=45)
                axes[i].set_ylabel(metric)
            st.pyplot(fig)
        else:
            st.warning("No models available for comparison.")

    else:
        # Single model forecast and display
        models = None
        scalers = None
        if model_type == "ARIMA":
            models, scalers = arima_models, arima_scalers
        elif model_type == "LSTM":
            models, scalers = lstm_models, lstm_scalers
        else:
            models, scalers = autots_models, {}

        if st.button("Generate Forecast"):
            with st.spinner("Generating predictions..."):
                try:
                    series = df[currency].dropna()
                    test_index, test_denorm, preds_denorm, rmse, mae, mape, upper, lower = run_forecast(model_type, base_name, test_size, series, models, scalers)

                    current = test_denorm[-1]
                    predicted = preds_denorm[-1]

                    st.header(f"{model_type} Forecast Results: {currency}")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Last Actual Rate", f"{current:.4f}")
                    col2.metric("Last Predicted Rate", f"{predicted:.4f}", delta=f"{predicted - current:.4f}")
                    col3.metric("RMSE", f"{rmse:.4f}")
                    col4.metric("MAE", f"{mae:.4f}")
                    st.metric("MAPE", f"{mape:.2f}%")

                    # Plot results
                    fig, ax = plt.subplots(figsize=(14, 6))
                    ax.plot(test_index, test_denorm, label='Actual', marker='o', color='blue')
                    ax.plot(test_index, preds_denorm, label='Predicted', linestyle='--', marker='x', color='orange')

                    if model_type == "AutoTS" and upper is not None and lower is not None:
                        ax.fill_between(test_index, lower[series.name], upper[series.name], alpha=0.2)

                    ax.set_title(f"{currency} {model_type} Forecast")
                    ax.legend()
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Forecasting failed: {str(e)}")

if __name__ == "__main__":
    main()