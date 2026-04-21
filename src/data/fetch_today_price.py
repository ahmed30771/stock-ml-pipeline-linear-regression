import requests
import pandas as pd


def get_today_price(symbol: str, limit=1, timeframe="1d"):
    if (limit > 100) or (limit == 0):
        raise ValueError("Limit cannot be greater than 100 or equal to zero")

    url = f"https://psxterminal.com/api/klines/{symbol}/{timeframe}?limit={int(limit)}"

    response = requests.get(url)
    data = response.json()

    rows = []
    if data["success"]:
        for candle in data["data"]:
            rows.append({
                "Date": candle["timestamp"],
                "Open": candle["open"],
                "High": candle["high"],
                "Low": candle["low"],
                "Close": candle["close"],
                "Volume": candle["volume"]
            })
        df = pd.DataFrame(rows)
        df["Date"] = pd.to_datetime(df["Date"], unit="ms")

        return df

    else:
        raise Exception("API Error")



