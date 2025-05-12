# Machine_Learning_Forex_Exchange_Rate_Predictor

## Table of Contents
* Background
* Exploratory Data Analysis, Data Preprocessing and Data Cleaning
* Building the Arima Model
* Building the LSTM Model
* The FrontEnd and StreamLit
* Discussion and Future Work

## Background
This project, part of a series on machine learning deployment, evaluates traditional and advanced models for forecasting currency forex pair trends. I compared **ARIMA, LSTM, AutoTS, and FBProphet-models** ideal for handling temporal patterns and seasonality. An interactive **Streamlit web app** was built to visualize results and compare performance metrics **(RMSE, MAPE, MAE)** across currency pairs, aiding model selection. Streamlit was chosen for its simplicity in creating Python-based web interfaces.

##  Exploratory Data Analysis, Data Preprocessing and Data Cleaning
In this machine learning project, Exploratory Data Analysis (EDA), data preprocessing, and data cleaning are vital for success. By thoroughly understanding and cleaning the data, and applying optimal preprocessing techniques, I establish a strong foundation for accurate predictions. These steps help identify patterns, address anomalies, and format the data (e.g., currencies as float, time as datetime) to enhance model performance. Additional tasks include trend analysis and checking data distribution for outliers and skewness.

## Building the Arima Model
For the ARIMA model, I checked for **stationarity** to help identify trends and relationships in real-world data. First-order differencing was applied, confirming stationarity of the currencies. This was validated by the **Augmented Dickey-Fuller (ADF) test**, where the **P-value was significantly below 0.05**, leading to rejection of the null hypothesis. Currencies were then normalized using **MinMax Scalers**. **ACF and PACF** plots further confirmed stationarity, showing autocorrelations dropping quickly to zero after lag 0. The differenced dataset, saved as Arima_df, was used exclusively for ARIMA.

The model was built with auto-arima, using a defined range of hyperparameters to determine the best fit for each currency.
    model = pm.auto_arima(
        train_scaled,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        max_p=5,
        d=0,
        max_d=0,
        max_q=5,
        information_criterion='aic'and saved

Since first-order differencing was already applied, differencing is set to zero. The parameters max_p and max_q represent the order of the autoregressive (AR) and moving average (MA) components of the ARIMA model, respectively. Before training, the dataset was split, using the last 60 days of the time series as the test set. The model was fitted only on the training data, not the test set. After training, forecasting was performed on the entire dataset, including the test data. The test data was transformed using the same scaling as the training data. Finally, predictions were denormalized and differencing was reversed to match expected currency values.

## Building the LSTM Model
For the LSTM model, stationarity is not required, but normalization using MinMax Scaler is essential due to the model's sensitivity to input data scale. Training LSTM took significantly longer than other models. I used Keras Tuner to identify the best hyperparameters, which were automatically applied to model building and forecasting. As with the ARIMA model, the LSTM was fitted only on training data, with transformations applied to train, test, and validation splits. Finally, denormalization was performed to restore currency values to their original scale, though reverse differencing was not needed.

##  The FrontEnd and StreamLit
The Streamlit app is primarily written in Python. To run it, fork the repository, ensure all required dependencies are installed on your local machine, and execute the command streamlit run main.py. You can then access the app via localhost:8501 in your browser.

## Results, Discussion and Future Work
Results from comparing RMSE, MAE, and MAPE scores reveal that the LSTM model outperforms ARIMA, AutoTS, and FBProphet, likely due to its ability to capture complex, non-linear patterns and long-term dependencies in time series data. LSTM ranked best for 11 currencies, second-best for 7, and worst for 4. However, ARIMA excelled with some currencies, ranking best for 8, second-best for 5, and worst for 9. AutoTS performed best for 3 currencies, second-best for 9, and worst for 10.

Given the varying performance of the models across different currencies, selecting the appropriate model depends on the specific currency pair and forecasting needs. The Streamlit app provides an intuitive interface to compare metrics and visualize results, helping users make informed decisions. For future improvements, I am considering incorporating ensemble methods to combine the strengths of multiple models  and also expanding the dataset to include additional economic indicators for more robust predictions.

## Potential Applications
**1. Healthcare:** Health tech companies can apply time series analysis for early detection of respiratory complications by monitoring heart and respiratory rates. This enables preventative care through early identification of issues, achieving significant sensitivity while minimizing false alarms.

**2. Financial Sector:** Financial firms can utilize this time series for market risk management, specifically in Value at Risk (VaR) modeling. By analyzing historical financial data such as stock prices or market volatility, they quantify risks and make informed investment decisions to mitigate potential losses.

**3. Retail Sector:** Retail stores can employ time series analysis for demand forecasting and inventory management. By predicting future demand trends from historical sales data, they maintain optimal stock levels, reduce stockouts, and enhance customer satisfaction.

**4. Energy Sector:** Energy companies can leverage this time series for electricity demand forecasting, which is crucial for grid stability and energy distribution planning. Accurate predictions ensure a reliable electricity supply and prevent blackouts by balancing supply and demand.

**5. Technology Sector:** Streaming services can leverage this time series analysis to predict user engagement and content popularity. This informs their recommendation algorithms and strategic decisions, driving subscriber growth and retention.


