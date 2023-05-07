# Foreign-Exchange-Value-Prediction
## Abstract
This project aims to analyze and model the EUR/USD exchange rate data to make predictions about future exchange rates using time series analysis. The project aims to compare several time series and deep learning models to find the best or most suitable model for this purpose. The scope of the project includes collecting daily exchange rate data from January 2004 to the current date, analyzing the data using time series analysis techniques such as time series decomposition, autocorrelation, and partial autocorrelation plots, selecting an appropriate time series model for the data, validating the model using statistical metrics such as rooted mean squared error (RMSE) and mean average error (MAE), and using the selected time series model to make predictions and forecast future EUR/USD exchange rates. Several tests on different time series models like AR, MA, ARIMA, SARIMAX, ETS, etc. were compared with deep learning models such as GRU, LSTM, and Bi-LSTM. Here, the results showcase that deep learning models outperformed time series models by a huge margin, where the best fitting model was LSTM with 0.0094 and 0.0071 RMSE and MAE, respectively. This project has the potential to provide valuable insights into global economic activity and help investors make informed decisions.

**Keywords:** AR, ARIMA, Bi-LSTM, ETS, Forecasting, Foreign Exchange Rate, GRU, LSTM, MA, SARIMAX, Time Series Analysis, USD/EUR

## 1. Introduction
Time series analysis is a statistical technique used to analyze and model data that varies over time. In this project, we will use time series analysis to analyze and model the EUR/USD foreign exchange price data. The EUR/USD exchange rate is one of the most widely traded currency pairs in the world and is therefore an important benchmark for global economic activity.

### 1.1. Problem Statement
The foreign exchange market is one of the largest and most liquid financial markets in the world. One of the most widely traded currency pairs is the EUR/USD exchange rate. Analyzing the fluctuations in the EUR/USD exchange rate can provide valuable insights into global economic activity and help investors make informed decisions. However, analyzing time series data can be challenging due to the complex patterns and trends in the data. The objective of this project is to use time series analysis to analyze and model the EUR/USD exchange rate data to make predictions about future exchange rates. Moreover, the project aims to compare several time series and deep learning models and find the best or most suitable model for this project.

### 1.2. Project Scope
The scope of this project includes the following:
**Data collection:** Collecting daily EUR/USD exchange rate data from January 2004 to the current date from Yahoo Finance.
**Data analysis:** Analyzing the data using time series analysis techniques such as time series decomposition, autocorrelation, and partial autocorrelation plots.
**Model selection and validation:** Selecting an appropriate time series model for the data and validating the model using statistical metrics such as mean squared error (MSE) and root mean squared error (RMSE).
**Forecasting:** Using the selected time series model to make predictions and forecast future EUR/USD exchange rates.

The project is implemented using the Python programming language and the following libraries: pandas, numpy, matplotlib, pmdarima, Plotly, scikit-learn, statsmodels, TensorFlow and yfinance. The project will be conducted in four phases: data collection and preparation, data analysis, model selection and validation, and forecasting and reporting.

## 2. Methodology
### 2.1. Data Description
The data used in this project is the daily EUR/USD exchange rate from January 2004 to real time. The data was obtained from the Federal Reserve Economic Data (FRED) database. The dataset consists of 5,012 observations, and each observation includes the date and the EUR/USD exchange rate like Open, High, Low, Close, Adj Close and Volume.

### 2.2. Data Preprocessing
The data underwent several preprocessing techniques, like scaling, stationary testing, and differencing methods for making the data stationary. Then, the data was further split into two parts, training and testing, where 80% of the data was used for training, and the rest for testing.

### 2.3. Model Creation
Further, the data was trained on several models. Some of them were popular time series models, and others were deep learning neural network models. We used autoregressive, moving average, ARIMA, SARIMAX, ETS, GRU, LSTM, and Bi-LSTM models to train the data.

**Autoregressive (AR) model:** The AR model is a time series model that predicts the next value in a time series based on past values. The model assumes that the next value in the time series is a linear combination of past values.

**Moving Average (MA) model:** The MA model is a time series model that predicts the next value in a time series based on the errors (the difference between the predicted and actual values) of past predictions. The model assumes that the errors are a linear combination of past errors.

**Autoregressive Integrated Moving Average (ARIMA) model:** The ARIMA model is a combination of the AR and MA models. It is a time series model that includes three components: the autoregressive component (AR), the moving average component (MA), and the integrated component (I), which represents the differencing operation required to make the time series stationary.

**Seasonal Autoregressive Integrated Moving Average with Exogenous Regressors (SARIMAX) model:** The SARIMAX model is an extension of the ARIMA model that includes additional regressors (exogenous variables) to account for the influence of external factors on the time series. The model also includes a seasonal component to account for periodic fluctuations in the data.

**Exponential Smoothing (ETS) model:** The ETS model is a time series model that uses a smoothing factor to calculate the next value in a time series based on past values. The model assumes that the next value is a weighted average of past values, with more recent values given greater weight.

**Gated Recurrent Unit (GRU) model:** The GRU model is a type of recurrent neural network (RNN) that is commonly used for time series analysis and forecasting. The model includes gating mechanisms that allow it to selectively update and forget information from past time steps.

**Long Short-Term Memory (LSTM) model:** The LSTM model is another type of RNN that is commonly used for time series analysis and forecasting. The model includes memory cells that allow it to retain information from past steps and selectively update and forget information.

**Bidirectional LSTM (Bi-LSTM) model:** The Bi-LSTM model is an extension of the LSTM model that includes two LSTM layers: one that processes the time series forward and one that processes the time series backward. This allows the model to capture both past and future dependencies in the data.

Finally, the best model underwent fine-tuning to give the desired results.

### 2.5. Evaluation Metrics
Several evaluation metrics, like R2 score, MSE, RMSE, and MAE, were used to evaluate all the models and select the best-performing model.


### 2.6. Forecasting
All of the models were used to forecast the price for the next 2 years, or 750 days. We’ve shown some figures below for the forecast we were able to achieve with different models.

## 3. Results
All the evaluation metrics from all the models are represented in the table below:

| Models  | RSME  | MAE |
| :------------ |:---------------:| -----:|
|Autoregressive | 0.0542 | 0.0445 |
|Moving Average (MA100) | 0.0430 | 0.0317 |
|ARIMA(1,0,2) | 0.0534 | 0.0435 |
|SARIMAX(1,0,2)x(1,1,[1],12) | 0.0726 | 0.0607 |
|ETS | 0.0512 | 0.0553 |
|GRU | 0.0138| 0.0108 |
| LSTM | 0.0094 | 0.0071 |
|Bi-LSTM | 0.0132 | 0.0106 |

According to our analysis, deep learning models like LSTM, Bi-LSTM, and GRU outperformed all of the time series techniques like AR, MA, ARIMA, SARIMAX, and ETS. Also, according to various tests, we can see that stationary tests and making the data stationary over time help significantly improve the accuracy, especially in time series models. But, the deep learning models do not require any such preprocessing of the data.

### 4. Conclusion 
In conclusion, the LSTM model gave the best test with an RMSE of 0.0094, with Bi-LSTM and GRU coming in close behind with 0.0138 and 0.0132 RMSE, respectively. Also, the result showcases that the use of deep learning models like LSTM, Bi-LSTM, and GRU are significantly better for foreign exchange price prediction than time series models such as AR, ARIMA, SARIMA, etc. Further, this project showcases that these models can be used for actual prediction and for foreseeing foreign exchange rates with such high results.

### 5. Future Work
To further expand this project, we can create models to predict and forecast the opening and closing prices of all forex pairs like AUD/USD, GBP/USD, etc. Moreover, we can aim to improve the accuracy of models through hyperparameter optimization. To make these models more accessible, we plan to create an interactive web app that not only shows descriptions of the dataset but also the important visualisation required for trading, like candlestick graphs. It will also give the user the option to choose from all the models we have created on the basis of evaluation metrics selected by the user. In the future, we also plan to include more sophisticated models like the prophet and neural prophet from Meta.
