import pandas_ta as ta
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pickle
import json
import os
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from fbprophet import Prophet
import statsmodels.api as sm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

st.set_page_config(page_title="Dashboard des Actifs Financiers", layout="wide")

@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def save_model_coefficients(model, model_name, output_dir="model_coefficients"):
    os.makedirs(output_dir, exist_ok=True)
    coefficients = {}
    if hasattr(model, "coef_"):
        coefficients["coefficients"] = model.coef_.tolist()
    if hasattr(model, "intercept_"):
        coefficients["intercept"] = model.intercept_.tolist()
    if hasattr(model, "params"):
        coefficients["parameters"] = model.params.to_dict()
    json_path = os.path.join(output_dir, f"{model_name}.json")
    with open(json_path, "w") as f:
        json.dump(coefficients, f, indent=4)
    pickle_path = os.path.join(output_dir, f"{model_name}.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Coefficients saved for {model_name} in {output_dir}")

data_files = {
    'S&P 500': 'indicateurs_economique/sp500_with_indicators.csv',
    'Bitcoin': 'indicateurs_economique/bitcoin_historical_data_cleaned.csv',
    'Gold': 'indicateurs_economique/gold_historical_data_cleaned.csv'
}

with st.sidebar:
    st.header("Sélection de l'Actif")
    selected_asset = st.selectbox("Choisissez un actif :", list(data_files.keys()))

if selected_asset:
    df = load_data(data_files[selected_asset])
    min_date, max_date = df['Date'].min(), df['Date'].max()
    
    col_start, col_end = st.columns(2)
    with col_start:
        start_date = st.date_input("Début", min_date, min_value=min_date, max_value=max_date)
    with col_end:
        end_date = st.date_input("Fin", max_date, min_value=min_date, max_value=max_date)
    df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
    
    df['Returns'] = df['Close'].pct_change()
    df['Annual_Return'] = df['Returns'].mean() * 252
    df['Annual_Volatility'] = df['Returns'].std() * np.sqrt(252)
    
    tabs = st.tabs(["Overview", "Détails", "Comparaisons", "Prédiction"])
    
    with tabs[0]:
        st.header(f"Aperçu de {selected_asset}")
        st.metric("Rendement Annuel", f"{df['Annual_Return'].iloc[-1]:.2%}")
        st.metric("Volatilité Annuelle", f"{df['Annual_Volatility'].iloc[-1]:.2%}")
    
    with tabs[1]:
        st.header(f"Détails de {selected_asset}")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Chandeliers"))
        for indicator in ["SMA", "EMA", "MACD", "RSI"]:
            if indicator in df.columns:
                fig.add_trace(go.Scatter(x=df['Date'], y=df[indicator], mode='lines', name=indicator))
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        st.header("Comparaison entre actifs (Transformation Logarithmique)")
        selected_assets = st.multiselect("Sélectionnez les actifs", list(data_files.keys()), default=list(data_files.keys()))
        fig = go.Figure()
        for asset in selected_assets:
            temp_df = load_data(data_files[asset])
            temp_df = temp_df[(temp_df['Date'] >= pd.to_datetime(start_date)) & (temp_df['Date'] <= pd.to_datetime(end_date))]
            temp_df['Log_Close'] = np.log(temp_df['Close'])
            fig.add_trace(go.Scatter(x=temp_df['Date'], y=temp_df['Log_Close'], mode='lines', name=f"{asset} (Log)"))
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[3]:
        st.header("Prédiction des Prix sur 50 Jours")
        pred_days = 50
        model_choice = st.selectbox("Sélectionner un modèle de prédiction", ["Linear Regression", "Prophet", "Logistic Regression"])
        
        if model_choice == "Linear Regression":
            model = LinearRegression()
        elif model_choice == "Prophet":
            model = Prophet()
        else:
            model = sm.Logit()
        
        X = np.array(df['Close']).reshape(-1, 1)
        y = np.array(df['Close'].shift(-pred_days)).reshape(-1, 1)
        model.fit(X, y)
        predictions = model.predict(X)
        
        df['Predicted_Close'] = np.nan
        df.iloc[:-pred_days, df.columns.get_loc('Predicted_Close')] = predictions.flatten()
        
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name="Réel"))
        fig_pred.add_trace(go.Scatter(x=df['Date'], y=df['Predicted_Close'], mode='lines', name="Prédit"))
        st.plotly_chart(fig_pred, use_container_width=True)
        
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        st.metric("MSE", f"{mse:.2f}")
        st.metric("RMSE", f"{rmse:.2f}")
