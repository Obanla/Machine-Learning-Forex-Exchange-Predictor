import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- Configuration ---
MODELS_DIR = "Arima_Models"
SCALERS_DIR = "Arima_Scalers"


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

def inverse_diff(diffs, initial):
    restored = [initial + diffs[0]]
    for i in range(1, len(diffs)):
        restored.append(restored[i - 1] + diffs[i])
    return np.array(restored)

# Add this MAPE calculation function
def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0  # Avoid division by zero
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100


def main():
    st.set_page_config(page_title="Forex Predictor", page_icon="📈", layout="wide")
    st.title("💹 Forex Exchange Rate Predictor")

    # Load models and scalers
    models, scalers = {}, {}
    if os.path.exists(MODELS_DIR):
        models = {f.replace("model_", "").replace(".joblib", ""): joblib.load(os.path.join(MODELS_DIR, f))
                  for f in os.listdir(MODELS_DIR) if f.startswith("model_")}
    if os.path.exists(SCALERS_DIR):
        scalers = {f.replace("scaler_", "").replace(".joblib", ""): joblib.load(os.path.join(SCALERS_DIR, f))
                   for f in os.listdir(SCALERS_DIR) if f.startswith("scaler_")}

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

    # Get model and scaler
    model_key = CURRENCY_MAPPING.get(currency)
    if not model_key or model_key not in models:
        st.error(f"No model available for {currency}")
        return

    model = models[model_key]
    scaler = scalers.get(model_key)
    if not scaler:
        st.error(f"No scaler found for {currency}")
        return

    d = 1  # Differencing order

    st.header(f"Actual vs Predicted: {currency}")

    #--- Historical Data Display ---
    st.header(f"Historical Exchange Rates: {currency}")
    fig, ax = plt.subplots(figsize=(12, 6))
    df[currency].plot(ax=ax, title="Historical Data")
    st.pyplot(fig)


    if st.button("Generate Test Set Forecast"):
        with st.spinner("Generating predictions..."):
            try:
                series = df[currency].dropna()
                train, test = series.iloc[:-test_size], series.iloc[-test_size:]

                # Scale train and test on the original (not differenced) series
                train_scaled = scaler.transform(train.values.reshape(-1, 1)).flatten()
                test_scaled = scaler.transform(test.values.reshape(-1, 1)).flatten()

                # Forecasting loop (walk-forward)
                history = list(train_scaled)
                predictions_scaled = []
                for t in range(len(test_scaled)):
                    fitted = model.fit(history)
                    yhat = fitted.predict()[0]
                    predictions_scaled.append(yhat)
                    history.append(test_scaled[t])

                # Inverse scaling
                preds_denorm = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()
                test_denorm = scaler.inverse_transform(np.array(test_scaled).reshape(-1, 1)).flatten()

                current = test_denorm[-1]  # Get last actual value
                predicted = preds_denorm[-1]  # Get last predicted value

                # New error metrics
                rmse = np.sqrt(mean_squared_error(test_denorm, preds_denorm))
                mae = mean_absolute_error(test_denorm, preds_denorm)
                mape = calculate_mape(test_denorm, preds_denorm)

                # Display metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Last Actual Rate", f"{current:.4f}")
                col2.metric("Last Predicted Rate", f"{predicted:.4f}",
                            delta=f"{predicted - current:.4f}")
                col3.metric("RMSE", f"{rmse:.4f}")
                col4.metric("MAE", f"{mae:.4f}")

                # Display MAPE in its own row
                st.metric("MAPE", f"{mape:.2f}%")

                # If you did NOT difference the data before scaling/modeling, do NOT apply inverse_diff!
                # Just plot preds_denorm vs test_denorm directly:
                fig, ax = plt.subplots(figsize=(14, 6))
                ax.plot(test.index, test_denorm, label='Actual', marker='o', color='blue')
                ax.plot(test.index, preds_denorm, label='Predicted', linestyle='--', marker='x', color='orange')
                ax.set_title(f"{currency} Forecast (Original Scale)")
                ax.legend()
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Forecasting failed: {str(e)}")

if __name__ == "__main__":
    main()