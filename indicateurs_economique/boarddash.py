import dash
from dash import dcc, html
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf

# Initialiser l'application Dash
app = dash.Dash(__name__)

# Télécharger les données financières
tickers = {"S&P 500": "^GSPC", "Bitcoin": "BTC-USD", "Gold": "GC=F"}
data = {name: yf.download(ticker, start="2019-01-01", end="2025-02-11") for name, ticker in tickers.items()}

# Créer des figures
fig_performance = go.Figure()
for name, df in data.items():
    fig_performance.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode='lines', name=name))
fig_performance.update_layout(title="Comparaison des performances", xaxis_title="Date", yaxis_title="Prix")

# Graphique en chandeliers pour le S&P 500
fig_candlestick = go.Figure(data=[go.Candlestick(x=data['S&P 500'].index,
                                                 open=data['S&P 500']['Open'],
                                                 high=data['S&P 500']['High'],
                                                 low=data['S&P 500']['Low'],
                                                 close=data['S&P 500']['Close'],
                                                 name='S&P 500')])
fig_candlestick.update_layout(title="Graphique en chandeliers - S&P 500")

# Disposition du dashboard
app.layout = html.Div(children=[
    html.H1(children="Dashboard Marché Financier"),
    dcc.Graph(id='performance', figure=fig_performance),
    dcc.Graph(id='candlestick', figure=fig_candlestick)
])

# Exécuter l'application
if __name__ == '__main__':
    app.run_server(debug=True)
