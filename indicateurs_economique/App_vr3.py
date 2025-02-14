
import pandas_ta as ta
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import mplfinance as mpf
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Configuration de la page
st.set_page_config(page_title="Dashboard des Actifs Financiers", layout="wide")

# Fonction pour charger les données en cache
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# Liste des fichiers de données
data_files = {    
    'S&P 500': 'indicateurs_economique\sp500_with_indicators.csv',
    'Bitcoin': 'indicateurs_economique/bitcoin_historical_data_cleaned.csv',
    'Gold': 'indicateurs_economique\gold_historical_data_cleaned.csv'
}

# Barre latérale pour la sélection de l'actif
with st.sidebar:
    st.header("Sélection de l'Actif")
    selected_asset = st.selectbox("Choisissez un actif :", list(data_files.keys()))

if selected_asset:
    df = load_data(data_files[selected_asset])

    # Sélection des dates sur une même ligne
    min_date, max_date = df['Date'].min(), df['Date'].max()
    col_start, col_end = st.columns(2)
    with col_start:
        start_date = st.date_input("Début", min_date, min_value=min_date, max_value=max_date)
    with col_end:
        end_date = st.date_input("Fin", max_date, min_value=min_date, max_value=max_date)
    df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]

    # Calcul des indicateurs techniques
    df['SMA'] = df['Close'].rolling(window=20).mean()
    df['EMA'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['MACD'] = ta.macd(df['Close'], fast=12, slow=26, signal=9)['MACD_12_26_9']
    df['MDD'] = (df['Close'] / df['Close'].cummax() - 1) * 100
    df['Sharpe_Ratio'] = df['Close'].pct_change(fill_method=None).mean() / df['Close'].pct_change(fill_method=None).std()
    bollinger = ta.bbands(df['Close'], length=20, std=2)
    df['BB_Upper'] = bollinger['BBU_20_2.0']
    df['BB_Middle'] = bollinger['BBM_20_2.0']
    df['BB_Lower'] = bollinger['BBL_20_2.0']

    # Création des onglets
    tabs = st.tabs(["Overview", "Détails", "Comparaisons", "Prédiction"])

    # Onglet Aperçu
    with tabs[0]:
        st.header(f"Aperçu de {selected_asset}")
        rendement_moyen = df['Close'].pct_change().mean()
        volatilite = df['Close'].pct_change().std()
        tendance = "Hausse" if df['Close'].iloc[-1] > df['Close'].iloc[0] else "Baisse"
        
        max_close = df['Close'].max()
        max_close_date = df[df['Close'] == max_close]['Date'].iloc[0]
        min_close = df['Close'].min()
        min_close_date = df[df['Close'] == min_close]['Date'].iloc[0]
        
        st.metric("Rendement Moyen", f"{rendement_moyen:.2%}")
        st.metric("Volatilité", f"{volatilite:.2%}")
        st.metric("Tendance du marché", tendance)
        st.metric("Close Max", f"{max_close:.2f} ({max_close_date.date()})")
        st.metric("Close Min", f"{min_close:.2f} ({min_close_date.date()})")


    # Onglet Détails
    with tabs[1]:
        st.header("Détails de l'Actif")
        df_candle = df.set_index('Date')
        fig, ax = mpf.plot(df_candle, type='candle', style='charles', volume=True, returnfig=True)
        st.pyplot(fig)
        
        selected_indicators = st.multiselect("Sélectionnez les indicateurs à afficher", ["SMA", "EMA", "RSI", "MACD", "MDD", "Sharpe_Ratio", "BB_Upper", "BB_Middle", "BB_Lower"])
        
        for indicator in selected_indicators:
            st.subheader(f"Indicateur : {indicator}")
            st.line_chart(df[['Date', indicator]].set_index('Date'))

    # Onglet Comparaisons
    with tabs[2]:
        st.header("Comparaison entre actifs")
        assets_to_compare = st.multiselect(
            "Sélectionnez les actifs à comparer", 
            options=list(data_files.keys()), 
            default=list(data_files.keys())
        )
        
        if assets_to_compare:
            fig_prices = go.Figure()
            fig_volume = go.Figure()
            
            for asset in assets_to_compare:
                temp_df = load_data(data_files[asset])
                temp_df = temp_df[(temp_df['Date'] >= pd.to_datetime(start_date)) & (temp_df['Date'] <= pd.to_datetime(end_date))]
                
                # Normalisation des prix à une échelle de 1000
                temp_df['Normalized_Close'] = (temp_df['Close'] - temp_df['Close'].min()) / (temp_df['Close'].max() - temp_df['Close'].min()) * 1000
                
                # Normalisation des volumes à une échelle de 100
                temp_df['Normalized_Volume'] = (temp_df['Volume'] - temp_df['Volume'].min()) / (temp_df['Volume'].max() - temp_df['Volume'].min()) * 100
                
                # Ajout des courbes
                fig_prices.add_trace(go.Scatter(x=temp_df['Date'], y=temp_df['Normalized_Close'], mode='lines', name=f"{asset} (Normalisé)"))
                fig_volume.add_trace(go.Scatter(x=temp_df['Date'], y=temp_df['Normalized_Volume'], mode='lines', name=f"Volume {asset} (Normalisé)"))
            
            fig_prices.update_layout(title="Évolution des prix des actifs (Normalisé à 1000)", xaxis_title="Date", yaxis_title="Prix Normalisé")
            fig_volume.update_layout(title="Volume échangé sur le marché (Normalisé à 100)", xaxis_title="Date", yaxis_title="Volume Normalisé")
            
            st.plotly_chart(fig_prices, use_container_width=True)
            st.plotly_chart(fig_volume, use_container_width=True)
        else:
            st.warning("Veuillez sélectionner au moins un actif pour la comparaison.")
    # Onglet Prédiction
    with tabs[3]:
        st.header("Prédiction des Prix")
        col_pred_start, col_pred_end = st.columns(2)
        with col_pred_start:
            pred_start_date = st.date_input("Début de la prédiction", min_date, key="pred_start", min_value=min_date, max_value=max_date)
        with col_pred_end:
            pred_end_date = st.date_input("Fin de la prédiction", max_date, key="pred_end", min_value=min_date, max_value=max_date)
        prediction_model = st.selectbox("Choisissez un modèle de prédiction", ["ARIMA", "Prophet", ])
        if st.button("Lancer la prédiction"):
            st.info("Modèle en cours d'entraînement...")
            st.success(f"Prédiction terminée avec {prediction_model} !")
            st.metric("Accuracy", "{}")
            st.metric("Précision", "{}")

st.markdown("""
<style>
    .stApp { background-color: #f5f5f5; font-family: 'Arial', sans-serif; }
</style>
""", unsafe_allow_html=True)

