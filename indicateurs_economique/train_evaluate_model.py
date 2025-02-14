import pickle
import json
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from fbprophet import Prophet
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tbats import TBATS
from sklearn.metrics import mean_squared_error

def train_model(df, model_choice, pred_days=50, output_dir="model_coefficients"):
    os.makedirs(output_dir, exist_ok=True)
    df['Prediction'] = df['Close'].shift(-pred_days)
    df = df.dropna()
    X = np.array(df['Close']).reshape(-1, 1)
    y = np.array(df['Prediction']).reshape(-1, 1)
    
    if model_choice == "Linear Regression":
        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)
    elif model_choice == "Prophet":
        df_prophet = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        model = Prophet()
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=pred_days)
        forecast = model.predict(future)
        predictions = forecast['yhat'].values
    elif model_choice == "Logistic Regression":
        X = sm.add_constant(X)
        model = sm.Logit(y, X)
        model = model.fit()
        predictions = model.predict(X)
    elif model_choice == "Holt-Winters":
        model = ExponentialSmoothing(df['Close'], trend='add', seasonal='add', seasonal_periods=12).fit()
        predictions = model.forecast(pred_days)
    elif model_choice == "ARIMA":
        model = SARIMAX(df['Close'], order=(5,1,0)).fit()
        predictions = model.forecast(pred_days)
    elif model_choice == "TBATS":
        estimator = TBATS(seasonal_periods=12)
        model = estimator.fit(df['Close'])
        predictions = model.forecast(steps=pred_days)
    
    mse = mean_squared_error(y[:len(predictions)], predictions)
    rmse = np.sqrt(mse)
    
    model_path = os.path.join(output_dir, f"{model_choice}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    return model_path, mse, rmse
