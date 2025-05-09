import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Model Configuration
PROPHET_MODELS_DIR = "Prophet_Models"


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



def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100


def main():
    st.set_page_config(page_title="Forex Prophet Predictor", page_icon="📈", layout="wide")
    st.title("💹 Forex Exchange Rate Predictor with Prophet")

    # Load Prophet models
    prophet_models = {}

    if os.path.exists(PROPHET_MODELS_DIR):
        prophet_models = {f.replace("prophet_model_", "").replace(".joblib", ""): joblib.load(os.path.join(PROPHET_MODELS_DIR, f))
                 for f in os.listdir(PROPHET_MODELS_DIR) if f.startswith("prophet_model_")}

    # Load historical data
    try:
        df = pd.read_csv('df_Prophet_Autots.csv',
                         parse_dates=['Time Serie'],
                         index_col='Time Serie')
    except FileNotFoundError:
        st.error("Historical data file not found")
        return

    # Adding the UI Components
    currency = st.sidebar.selectbox("Select Currency", df.columns)
    test_size = st.sidebar.slider("Test Set Size (days)", 20, 120, 60)

    # Get base name from mapping
    base_name = CURRENCY_MAPPING.get(currency)
    if not base_name:
        st.error(f"No configuration found for {currency}")
        return

    # Get Prophet model
    model = prophet_models.get(base_name)
    if not model:
        st.error(f"No Prophet model available for {currency}")
        return

    st.header(f"Actual vs Predicted: {currency}")

    # Plot historical data
    fig, ax = plt.subplots(figsize=(12, 6))
    df[currency].plot(ax=ax, title="Historical Data")
    st.pyplot(fig)

    if st.button("Generate Test Set Forecast"):
        with st.spinner("Generating predictions..."):
            try:
                series = df[currency].dropna().reset_index()
                series.columns = ['ds', 'y']

                # Split data
                train = series.iloc[:-test_size]
                test = series.iloc[-test_size:]

                # Generate future dates
                future = pd.DataFrame({'ds': test['ds']})

                # Make prediction
                forecast = model.predict(future)

                # Plot results
                fig, ax = plt.subplots(figsize=(14, 6))
                ax.plot(test['ds'], test['y'], label='Actual', marker='o', color='blue')
                ax.plot(forecast['ds'], forecast['yhat'], label='Predicted', linestyle='--', marker='x', color='orange')
                ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2)
                ax.set_title(f"{currency} Forecast")
                ax.legend()
                st.pyplot(fig)

                # Calculate metrics
                current = test['y'].iloc[-1]
                predicted = forecast['yhat'].iloc[-1]

                rmse = np.sqrt(mean_squared_error(test['y'], forecast['yhat']))
                mae = mean_absolute_error(test['y'], forecast['yhat'])
                mape = calculate_mape(test['y'], forecast['yhat'])

                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Last Actual Rate", f"{current:.4f}")
                col2.metric("Last Predicted Rate", f"{predicted:.4f}", delta=f"{predicted - current:.4f}")
                col3.metric("RMSE", f"{rmse:.4f}")
                col4.metric("MAE", f"{mae:.4f}")
                st.metric("MAPE", f"{mape:.2f}%")

            except Exception as e:
                st.error(f"Forecasting failed: {str(e)}")


if __name__ == "__main__":
    main()
