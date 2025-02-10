import yfinance as yf
import pandas as pd

symbol = "GC=F"  

start_date = "2019-01-01"
end_date = "2025-02-10"

data = yf.download(symbol, start=start_date, end=end_date)

if data.empty:
    print("Erreur : Aucune donnée n'a été récupérée pour l'or.")
    exit()

output_file = "gold_historical_data_from_2019.csv"
data.to_csv(output_file)

print(f"Les données historiques de l'or depuis 2019 ont été sauvegardées dans {output_file}.")
print(data.head())


######
import requests
import pandas as pd

api_key = 'USPUXOQBZLMU75EH'

crypto_symbol = 'BTC'
market = 'USD'
url = f'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={crypto_symbol}&market={market}&apikey={api_key}'

response = requests.get(url)
if response.status_code != 200:
    print(f"Erreur : Impossible de récupérer les données (Code {response.status_code})")
    exit()

data = response.json()

if "Time Series (Digital Currency Daily)" not in data:
    print("Erreur : Les données historiques ne sont pas disponibles.")
    exit()

daily_data = data["Time Series (Digital Currency Daily)"]

df = pd.DataFrame.from_dict(daily_data, orient='index')

print("Colonnes disponibles :", df.columns)

df = df.rename(columns={
    "1a. open (USD)": "Open Price (USD)",
    "2a. high (USD)": "High Price (USD)",
    "3a. low (USD)": "Low Price (USD)",
    "4a. close (USD)": "Close Price (USD)",
    "6. market cap (USD)": "Market Cap (USD)"
})

df.index = pd.to_datetime(df.index)
df.sort_index(inplace=True)

df = df[df.index >= "2019-01-01"]

output_file = "bitcoin_historical_data.csv"
df.to_csv(output_file)

print(df.head())
print(f"Les données historiques du Bitcoin depuis 2019 ont été sauvegardées dans {output_file}.")


########
import requests
import pandas as pd

api_key = "USPUXOQBZLMU75EH"

symbol = "XAUUSD"
url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}"

response = requests.get(url)

if response.status_code != 200:
    print(f"Erreur : Impossible de récupérer les données (Code {response.status_code})")
    exit()

data = response.json()

if "Time Series (Daily)" not in data:
    print("Erreur : Les données historiques ne sont pas disponibles.")
    exit()

daily_data = data["Time Series (Daily)"]
df = pd.DataFrame.from_dict(daily_data, orient="index")

df.columns = ["Open", "High", "Low", "Close", "Volume"]
df.index = pd.to_datetime(df.index)
df.sort_index(inplace=True)

df = df[df.index >= "2019-01-01"]

output_file = "gold_historical_data_from_2019.csv"
df.to_csv(output_file)

print(f"Les données historiques de l'or depuis 2019 ont été sauvegardées dans {output_file}.")
print(df.head())


######



