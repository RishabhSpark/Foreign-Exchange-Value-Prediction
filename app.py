import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers, callbacks
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from keras.models import load_model
import streamlit as st
from pandas_datareader import data as pdr
import yfinance as yf
import pmdarima as pm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import pacf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.ar_model import AutoReg
from datetime import date
# from datetime import timedelt
import plotly.express as px
import plotly.graph_objs as g

st.title("Forex Pair Price Prediction and Forecasting Using Time Series & Deep Learning Techinques")
start_date_input=st.date_input('Enter Start Date')
end_date_input=st.date_input('Enter End Date')
ticker_input = st.selectbox("Please Select Forex Pair",('EUR/USD', 'CHF/USD', 'AUD/USD','JPY/USD', 'GBP/USD', 'NZD/USD','INR/USD'))


yf.pdr_override()
df = pdr.get_data_yahoo(ticker_input, start=start_date_input, end=end_date_input, interval='1d')
st.write(df.describe())

model_input = st.selectbox("Please Select Desired Model for Prediction",('Moving Average', 'AutoRegression', 'ARIMA','SARIMA', 'ETS', 'VAR','Prophet','LSTM','BiLSTM','GRU','NeuralProphet'))
model_input2 = st.selectbox("Please Select Desired Model for Forecasting",('Moving Average', 'AutoRegression', 'ARIMA','SARIMA', 'ETS', 'VAR','Prophet','LSTM','BiLSTM','GRU','NeuralProphet'))
forecast_input=st.number_input('Enter Forecast Range')