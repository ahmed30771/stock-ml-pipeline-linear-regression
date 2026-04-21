from src import load_config, get_today_price, StockModelRunner
import pandas as pd


def predict_latest_price():
    # 1. Load lastest data
    new_df = get_today_price(load_config()["symbol"], limit=1)

    # 2. Prepare data
    SMR = StockModelRunner()
    X = SMR.prepare_features(new_df)

    # 3. Load model pipeline
    pipe = SMR.load_model_pipeline()

    # 4. Predict data
    pred = pipe.predict(X)
    direction = round((X["Close"].item() - pred.item()) /
                      X["Close"].item() * 100, 2)

    return {
        "price": round(pred.item(), 2),
        "direction": direction
    }


if __name__ == "__main__":
    from datetime import date
    p = predict_latest_price()
    tx = f"Tomorrow predicted price is {p["price"]} ({p["direction"]}%)"
    print(tx)
