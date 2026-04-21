import pandas as pd
import pandas_ta as ta
import os
from ..data import load_config


class StockFeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the feature engineer.

        Args
        -------
        df: pd.DataFrame
            Stock DataFrame with OHLCV data
        """
        self.df = df.copy()
        self.config = load_config()


    def pct_change(self) -> pd.DataFrame:
        """
        Create Returns Features (percentage change)

        Returns
        -------
        pd.DataFrame
            DataFrame with returns features added
        """
        self.df["pct_change"] = self.df["Close"].pct_change().round(2)
        return self.df


    def create_lag(self) -> pd.DataFrame:
        """
        Create Lag Features (previous day prices)

        Returns
        -------
        pd.DataFrame
            DataFrame with lag features added
        """
        periods = self.config.get("features", {}).get("lag_days", [1, 2, 3])

        for period in periods:
            self.df[f"Lag{period}"] = self.df["Close"].shift(periods=period)

        return self.df


    def create_ma(self) -> pd.DataFrame:
        """
        Create Moving Average Features

        Returns
        -------
        pd.DataFrame
            DataFrame with moving average features added
        """
        periods = self.config.get("features", {}).get("sma", [20, 50, 100])
        for period in periods:
            self.df[f'MA_{period}'] = ta.sma(
                self.df["Close"], length=period).round(2)
        return self.df


    def create_rsi(self) -> pd.DataFrame:
        """
        Calculate RSI (Relative Strength Index).

        Returns
        -------
        pd.DataFrame
            DataFrame with RSI features added
        """
        periods = self.config.get("features", {}).get("rsi", [14])
        for period in periods:
            self.df[f"RSI_{period}"] = ta.rsi(
                close=self.df["Close"], length=period)
        return self.df


    def create_target(self) -> pd.DataFrame:
        """
        Create Target Features

        Returns
        -------
        pd.DataFrame
            DataFrame with target features added
        """
        target_col = self.config.get("model", {}).get("target_column", "Close")
        self.df["Target"] = self.df[target_col].shift(periods=-1)
        return self.df


    def create_all_features(self) -> pd.DataFrame:
        """
        Create all features in single pipeline

        Returns
        -------
        pd.DataFrame
            DataFrame with all features
        """
        # Create Return Features (percentage change)
        self.pct_change()

        # Create Lag Features (previous day prices)
        self.create_lag()

        # Create Moving Averages Features
        self.create_ma()

        # Create RSI Features
        self.create_rsi()

        # Create Target Features
        self.create_target()

        # Drop NaN row that were created during feature engineering
        self.df.dropna(inplace=True)
        return self.df


    def save_processed_data(self, folder_path="./data/processed") -> str:
        """
        Save processed data in csv file

        Args
        -------
        folder_path: str, optional
            Folder path for saving the processed data file.<br>
            Defaults to "./data/processed"

        Returns
        -------
        str
            Confirmation message with the saved file path.
        """
        path = os.path.join(os.path.abspath(folder_path),
                            f"{self.config['symbol']}.csv")
        self.df.to_csv(path_or_buf=path, index=False)
        return f"Processed data is save at path: {path}"
