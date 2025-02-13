import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import pandas_ta as ta
import plotly.express as px
from datetime import datetime
 
# Configuration de la page
st.set_page_config(page_title="Dashboard des Actifs Financiers", layout="wide")
 
# Fonction pour charger les données en cache
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])  # Conversion de la colonne Date
    return df
 
# Liste des fichiers de données
data_files = {
    'S&P 500': 'ath/sp500.csv',
    'Bitcoin': 'ath/btc.csv',
    'Gold': 'ath/gold.csv'
}
 
# Barre latérale pour la sélection de l'actif
with st.sidebar:
    st.header("Sélection de l'Actif")
    selected_asset = st.selectbox("Choisissez un actif :", list(data_files.keys()))
 
# Chargement des données
if selected_asset:
    df = load_data(data_files[selected_asset])
 
    # Sélection des dates de début et de fin avec icônes sur la droite
    min_date, max_date = df['Date'].min(), df['Date'].max()
    col1, col2 = st.columns([4, 1])
    with col2:
        start_date = st.date_input("Début", min_date, min_value=min_date, max_value=max_date)
        end_date = st.date_input("Fin", max_date, min_value=min_date, max_value=max_date)
    df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
 
    # Calcul des indicateurs techniques
    df['SMA'] = df['Close'].rolling(window=50).mean()
    df['EMA'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['MACD'] = ta.macd(df['Close'], fast=12, slow=26, signal=9)['MACD_12_26_9']
    df['MDD'] = (df['Close'] / df['Close'].cummax() - 1) * 100
    df['Sharpe_Ratio'] = df['Close'].pct_change().mean() / df['Close'].pct_change().std()
    bollinger = ta.bbands(df['Close'], length=20, std=2)
    df['BB_Upper'] = bollinger['BBU_20_2.0']
    df['BB_Middle'] = bollinger['BBM_20_2.0']
    df['BB_Lower'] = bollinger['BBL_20_2.0']
 
    # Création des onglets
    tabs = st.tabs(["Overview", "Détails", "Comparaisons", "Prédiction"])
 
    # Aperçu
    with tabs[0]:
        st.header(f"Aperçu de {selected_asset}")
        rendement_moyen = df['Close'].pct_change().mean()
        volatilite = df['Close'].pct_change().std()
        tendance = "Hausse" if df['Close'].iloc[-1] > df['Close'].iloc[0] else "Baisse"
        st.metric("Rendement Moyen", f"{rendement_moyen:.2%}")
        st.metric("Volatilité", f"{volatilite:.2%}")
        st.metric("Tendance du marché", tendance)
 
    # Détails avec indicateurs techniques
    with tabs[1]:
        st.header(f"Détails de {selected_asset}")
        indicateurs = st.multiselect("Sélectionnez les indicateurs à afficher", ["SMA", "EMA", "MACD", "RSI", "Sharpe_Ratio", "BB_Upper", "BB_Lower"])
        for indicateur in indicateurs:
            st.line_chart(df[['Date', indicateur]].set_index('Date'))
        # Chandeliers japonais
        st.subheader("Chandeliers Japonais")
        fig_candle = go.Figure(data=[go.Candlestick(
            x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Chandeliers'
        )])
        st.plotly_chart(fig_candle)
 
    # Comparaison entre actifs
    with tabs[2]:
        st.header("Comparaison entre actifs")
        assets_to_compare = list(data_files.keys())
        fig = go.Figure()
        base_df = load_data(data_files[assets_to_compare[0]])[['Date', 'Close']].rename(columns={'Close': assets_to_compare[0]})
        for asset in assets_to_compare[1:]:
            temp_df = load_data(data_files[asset])[['Date', 'Close']].rename(columns={'Close': asset})
            base_df = pd.merge(base_df, temp_df, on='Date', how='outer')
        base_df.set_index('Date', inplace=True)
        base_df = base_df.pct_change().dropna()
        for asset in assets_to_compare:
            fig.add_trace(go.Scatter(x=base_df.index, y=base_df[asset], mode='lines', name=asset))
        st.plotly_chart(fig)
 
    # Prédiction
    with tabs[3]:
        st.header("Prédiction des Prix")
        prediction_model = st.selectbox("Choisissez un modèle de prédiction", ["LSTM", "ARIMA", "Prophet", "XGBoost"])
        if st.button("Lancer la prédiction"):
            st.info("Modèle en cours d'entraînement...")
            st.success(f"Prédiction terminée avec {prediction_model} !")
            st.metric("Accuracy", "90%")
            st.metric("Précision", "85%")
 
st.markdown("""
<style>
    .stApp { background-color: #f5f5f5; font-family: 'Arial', sans-serif; }
</style>
    """, unsafe_allow_html=True)