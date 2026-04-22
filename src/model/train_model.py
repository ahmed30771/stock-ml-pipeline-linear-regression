import os
import pickle
import pandas as pd
from ..data import load_config
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


class StockPriceModel:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.config: dict = load_config()

    def split_data(self):
        test_size = self.config.get('model', {}).get('test_size', 0.2)
        shuffle = self.config.get('model', {}).get('shuffle', False)

        # features
        X = self.df.drop(columns=["Target", "Date"])

        # target
        y = self.df['Target']

        # train test split (for time series shuffle=False)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            shuffle=shuffle
        )
        return X_train, X_test, y_train, y_test

    def model_pipeline(self, X_train, y_train):
        pipe = make_pipeline(
            StandardScaler(),
            LinearRegression()
        )
        pipe.fit(X_train, y_train)
        return pipe

    def save_pipeline(self, pipe, path="./model/linear_regression_pipeline.pkl"):
        """
        Save a pipeline to disk using pickle.

        Args
        -------
        pipe: object
            Trained pipeline object to save.
        path: str, optional
            File path for saving the pipeline.<br>
            Defaults to "./model/pipeline.pkl"

        Returns
        -------
        str
            Confirmation message with the saved file path.
        """
        path = os.path.abspath(path)
        with open(path, "wb") as f:
            pickle.dump(pipe, f)
        return f"Pipeline is saved at path: {path}"
