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
from train_evaluate_model import train_model
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

st.set_page_config(page_title="Dashboard des Actifs Financiers", layout="wide")

@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

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
        model_choice = st.selectbox("Sélectionner un modèle de prédiction", ["Linear Regression", "Prophet", "Logistic Regression"])
        
        model_path, mse, rmse = train_model(df, model_choice, pred_days=50)
        
        st.metric("MSE", f"{mse:.2f}")
        st.metric("RMSE", f"{rmse:.2f}")
        st.success(f"Modèle sauvegardé : {model_path}")
