import pandas as pd
from ..data import load_config


class DataPreprocesser:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the preprocessor

        Args
        -------
        df: pd.DataFrame
            DataFrame of stock data.
        """
        self.config = load_config()
        self.df = df.copy()

    def col_rename(self) -> pd.DataFrame:
        """
        Select and rename the columns of DataFrame with "Date", "Open", "High", "Low", "Close", "Volume".

        Returns
        -------
        pd.DataFrame
            DataFrame with renamed columns.
        """
        # Original Columns Names:
        col_date = self.config["data"]["columns"]["date"]
        col_open = self.config["data"]["columns"]["open"]
        col_high = self.config["data"]["columns"]["high"]
        col_low = self.config["data"]["columns"]["low"]
        col_close = self.config["data"]["columns"]["close"]
        col_volume = self.config["data"]["columns"]["volume"]

        # Select and Renaming Columns
        all_cols = {
            col_date: "Date",
            col_open: "Open",
            col_high: "High",
            col_low: "Low",
            col_close: "Close",
            col_volume: "Volume"
        }

        self.df = self.df[all_cols.keys()].rename(columns=all_cols)

        return self.df
        
    def convert_to_datetime(self) -> pd.DataFrame:
        """
        Convert the 'Date' column to datetime format and sort by date.

        Returns
        -------
        pd.DataFrame
            DataFrame with 'Date' column converted to datetime dtype and sorted chronologically.
        """
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.df = self.df.sort_values('Date')
        return self.df

    def remove_duplicate_rows(self) -> pd.DataFrame:
        """
        Remove duplicated rows from DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with duplicated rows removed.
        """
        self.df = self.df.drop_duplicates(keep='last')
        return self.df

    def remove_missing_values(self) -> pd.DataFrame:
        """
        Drop missing values in the dataset.

        Returns
        -------
        pd.DataFrame
            DataFrame with missing rows removed.
        """
        self.df = self.df.dropna()
        return self.df



def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to load stock data.

    Args
    -------
    df: pd.DataFrame
        DataFrame of stock data.

    Returns
    -------
    pd.DataFrame
        DataFrame with stock data
    """
    pp = DataPreprocesser(df)
    pp.col_rename()
    pp.convert_to_datetime()
    pp.remove_missing_values()
    pp.remove_duplicate_rows()
    return pp.df



