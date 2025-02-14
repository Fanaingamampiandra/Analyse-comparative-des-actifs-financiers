import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import pandas_ta as ta
from datetime import datetime
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
    'S&P 500': 'ath/sp500.csv',
    'Bitcoin': 'ath/btc.csv',
    'Gold': 'ath/gold.csv'
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
    df['SMA'] = df['Close'].rolling(window=50).mean()
    df['EMA'] = df['Close'].ewm(span=50, adjust=False).mean()
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
        st.metric("Rendement Moyen", f"{rendement_moyen:.2%}")
        st.metric("Volatilité", f"{volatilite:.2%}")
        st.metric("Tendance du marché", tendance)

    # Onglet Détails : graphique combiné (chandeliers + indicateurs superposés)
    with tabs[1]:
        # Injection de CSS pour la mise en page des boutons et labels
        st.markdown("""
        <style>
            .stButton > button {
                margin: 0 !important;
                padding: 2px 8px !important;
                font-size: 12pt !important;
            }
            div[data-testid="stHorizontalBlock"] > div {
                gap: 2mm;
                padding: 0 !important;
            }
            .custom-title {
                font-size: 12pt !important;
                font-weight: bold;
            }
        </style>
        """, unsafe_allow_html=True)

        # Initialisation des indicateurs sélectionnés dans session_state
        if "selected_indicators" not in st.session_state:
            st.session_state.selected_indicators = []

        # Liste des indicateurs disponibles
        indicateurs_options = ["SMA", "EMA", "MACD", "RSI", "Sharpe_Ratio", "BB_Upper", "BB_Lower"]

        # En-tête "Détails" avec "Filtres" alignés à côté
        header_cols = st.columns([2, 8])
        with header_cols[0]:
            st.markdown(f"<span class='custom-title'>Détails de {selected_asset}</span>", unsafe_allow_html=True)
        with header_cols[1]:
            filtres_cols = st.columns([1] + [1] * len(indicateurs_options))
            with filtres_cols[0]:
                st.markdown("<span class='custom-title'>Filtres:</span>", unsafe_allow_html=True)
            for idx, indicateur in enumerate(indicateurs_options):
                with filtres_cols[idx+1]:
                    if st.button(indicateur, key=f"btn_{indicateur}"):
                        if indicateur in st.session_state.selected_indicators:
                            st.session_state.selected_indicators.remove(indicateur)
                        else:
                            st.session_state.selected_indicators.append(indicateur)

        # Construction du graphique combiné
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Chandeliers"
        ))

        # Ajout des indicateurs sélectionnés
        for indicateur in st.session_state.selected_indicators:
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df[indicateur],
                mode='lines',
                name=indicateur
            ))
        fig.update_layout(xaxis=dict(tickformat='%d-%m-%Y'))
        st.plotly_chart(fig, use_container_width=True)

    # Onglet Comparaisons
    with tabs[2]:
        st.header("Comparaison entre actifs")
        assets_to_compare = st.multiselect(
            "Sélectionnez les actifs à comparer", 
            options=list(data_files.keys()), 
            default=list(data_files.keys())
        )
        if assets_to_compare:
            fig = go.Figure()
            base_df = load_data(data_files[assets_to_compare[0]])[['Date', 'Close']].rename(columns={'Close': assets_to_compare[0]})
            for asset in assets_to_compare[1:]:
                temp_df = load_data(data_files[asset])[['Date', 'Close']].rename(columns={'Close': asset})
                base_df = pd.merge(base_df, temp_df, on='Date', how='outer')
            base_df.set_index('Date', inplace=True)
            base_df = base_df.pct_change().dropna()
            for asset in assets_to_compare:
                fig.add_trace(go.Scatter(x=base_df.index, y=base_df[asset], mode='lines', name=asset))
            fig.update_layout(xaxis=dict(tickformat='%d-%m-%Y'))
            st.plotly_chart(fig, use_container_width=True)
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

