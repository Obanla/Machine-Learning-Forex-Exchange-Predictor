import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from autots import AutoTS

# --- Configuration ---
AUTOTS_MODEL_PATH = "autots_model.joblib"  # Update path to your saved model


def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0  # Avoid division by zero
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

def main():
    st.set_page_config(page_title="Forex AutoTS Predictor", page_icon="📈", layout="wide")
    st.title("💹 Forex Exchange Rate Predictor with AutoTS")

    # Load AutoTS model
    try:
        model = joblib.load(AUTOTS_MODEL_PATH)
    except FileNotFoundError:
        st.error("AutoTS model file not found")
        return

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

    st.header(f"Actual vs Predicted: {currency}")

    # Plot historical data
    fig, ax = plt.subplots(figsize=(12, 6))
    df[currency].plot(ax=ax, title="Historical Data")
    st.pyplot(fig)

    if st.button("Generate Test Set Forecast"):
        with st.spinner("Generating predictions..."):
            try:
                series = df[[currency]].dropna()
                full_size = len(series)
                train_size = full_size - test_size

                # AutoTS needs this formatting for validation
                train = series.iloc[:train_size]
                test = series.iloc[train_size:]

                # 1. Temporary model adjustment
                temp_model = model
                temp_model.forecast_length = test_size  # Set to test window

                # 2. Recreate validation using LSTM-style approach
                temp_model.fit_data(train)
                prediction = temp_model.predict()

                # 3. Align forecasts with test index
                forecast = prediction.forecast.set_index(test.index)

                upper = prediction.upper_forecast
                lower = prediction.lower_forecast

                # Plot results
                fig, ax = plt.subplots(figsize=(14, 6))
                ax.plot(test.index, test[currency], label='Actual', marker='o')
                ax.plot(forecast.index, forecast[currency], label='Predicted', linestyle='--')
                ax.fill_between(forecast.index, lower[currency], upper[currency], alpha=0.3)
                ax.set_title(f"{currency} Forecast with Prediction Interval")
                ax.legend()
                st.pyplot(fig)

                # Metrics calculation
                current = test[currency].iloc[-1]
                predicted = forecast[currency].iloc[-1]
                test_values = test[currency].values
                pred_values = forecast[currency].values

                # Calculate metrics (FIXED)
                mse = mean_squared_error(test_values, pred_values)
                rmse = np.sqrt(mse)  # Manual RMSE calculation
                mae = mean_absolute_error(test_values, pred_values)
                mape = calculate_mape(test_values, pred_values)

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