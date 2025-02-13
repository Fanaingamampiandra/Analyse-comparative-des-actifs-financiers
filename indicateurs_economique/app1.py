import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import pandas_ta as ta 
import plotly.express as px 

st.set_page_config(page_title="Dashboard des Actifs Financiers", layout="wide")

@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

data_files = {
    'S&P 500': 'ath/sp500.csv',
    'Bitcoin': 'ath/btc.csv',
    'Gold': 'ath/gold.csv'
}

with st.expander("Sélection de l'Actif", expanded=True):
    selected_asset = st.radio("Choisissez un actif :", list(data_files.keys()))

df = load_data(data_files[selected_asset]) 



df['RSI'] = ta.rsi(df['Close'], length=14)
df['MACD'] = ta.macd(df['Close'], fast=12, slow=26, signal=9)['MACD_12_26_9']
bollinger = ta.bbands(df['Close'], length=20, std=2)
df['BB_Upper'] = bollinger['BBU_20_2.0']
df['BB_Middle'] = bollinger['BBM_20_2.0']
df['BB_Lower'] = bollinger['BBL_20_2.0']

df['IBMA_200'] = df['Close'].rolling(window=200).mean()
df['IBMA_500'] = df['Close'].rolling(window=500).mean()
df['IBMA_1000'] = df['Close'].rolling(window=1000).mean()

tabs = st.tabs(["Overview", "Détails", "Comparaisons"])

with tabs[0]:
    st.header(f"Aperçu de {selected_asset}")
    st.write(df.describe())

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name=selected_asset))
    fig.update_layout(title=f"Évolution du prix de clôture de {selected_asset}",
                      xaxis_title="Date",
                      yaxis_title="Prix de clôture",
                      xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

with tabs[1]:
    st.header(f"Détails de {selected_asset}")
    
    detail_tabs = st.tabs(["RSI", "MACD", "Bandes de Bollinger"])
    
    
    with detail_tabs[0]:
        st.subheader("RSI (Relative Strength Index)")
        
        if 'RSI' in df.columns:
            st.write("Valeur actuelle du RSI :", df['RSI'].iloc[-1])
            
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], mode='lines', name='RSI'))
            fig_rsi.update_layout(title="Évolution du RSI",
                                  xaxis_title="Date",
                                  yaxis_title="RSI",
                                  xaxis_rangeslider_visible=True)
            st.plotly_chart(fig_rsi)
        else:
            st.warning("La colonne 'RSI' n'est pas disponible dans les données.")
    
    with detail_tabs[1]:
        st.subheader("MACD (Moving Average Convergence Divergence)")
        
        if 'MACD' in df.columns:
            st.write("Valeur actuelle du MACD :", df['MACD'].iloc[-1])
            
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], mode='lines', name='MACD'))
            fig_macd.update_layout(title="Évolution du MACD",
                                   xaxis_title="Date",
                                   yaxis_title="MACD",
                                   xaxis_rangeslider_visible=True)
            st.plotly_chart(fig_macd)
        else:
            st.warning("La colonne 'MACD' n'est pas disponible dans les données.")
    
    with detail_tabs[2]:
        st.subheader("Bandes de Bollinger (BB_Middle)")
        
        if 'BB_Middle' in df.columns:
            st.write("Valeur actuelle de la Bande Moyenne (BB_Middle) :", df['BB_Middle'].iloc[-1])
            
            fig_bb = go.Figure()
            fig_bb.add_trace(go.Scatter(x=df['Date'], y=df['BB_Upper'], mode='lines', name='BB Upper'))
            fig_bb.add_trace(go.Scatter(x=df['Date'], y=df['BB_Middle'], mode='lines', name='BB Middle'))
            fig_bb.add_trace(go.Scatter(x=df['Date'], y=df['BB_Lower'], mode='lines', name='BB Lower'))
            fig_bb.update_layout(title="Bandes de Bollinger",
                                 xaxis_title="Date",
                                 yaxis_title="Valeur",
                                 xaxis_rangeslider_visible=True)
            st.plotly_chart(fig_bb)
        else:
            st.warning("Les colonnes 'BB_Upper', 'BB_Middle' ou 'BB_Lower' ne sont pas disponibles dans les données.")

with tabs[2]:
    st.header("Comparaisons")
    st.write("Comparaison de l'actif sélectionné avec d'autres actifs.")
    assets_to_compare = st.multiselect("Sélectionnez les actifs à comparer :", list(data_files.keys()), default=[selected_asset])
    if assets_to_compare:
        fig = go.Figure()
        for asset in assets_to_compare:
            data = load_data(data_files[asset])
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name=asset))
        fig.update_layout(title="Comparaison des prix de clôture",
                          xaxis_title="Date",
                          yaxis_title="Prix de clôture",
                          xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
    else:
        st.warning("Veuillez sélectionner au moins un actif pour la comparaison.")

st.markdown("""
    <style>
    .stApp {
        background-color: #f5f5f5;
        font-family: 'Arial', sans-serif;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 18px;
        padding: 10px;
    }
    .stTabs [data-baseweb="tab"] > div {
        display: flex;
        align-items: center;
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)