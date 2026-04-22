import pickle
import os
import pandas as pd
from ..data import load_config, preprocess_data
from ..features import StockFeatureEngineer


class StockModelRunner:
    def __init__(self,
                 pipe_path="./model/linear_regression_pipeline.pkl",
                 ):
        self.pipe_path = os.path.abspath(pipe_path)
        self.config = load_config()
        self.old_df_path = os.path.join(os.path.abspath(
            self.config["data"]["file_path"]), f"{self.config["symbol"]}.csv")

    def prepare_features(self, new_df: pd.DataFrame):
        """
        Concatenate old & new DataFrame, preprocessing and feature engineering

        Args
        -------
        new_df: DataFrame
            Input DataFrame with the following columns:
            - `Date` (object): The specific time period the data refers to.
            - `Open` (int | float): Starting traded price.
            - `High` (int | float): Highest price of specific period.
            - `Low` (int | float): Lowest price of specific period.
            - `Close` (int | float): Final traded price.
            - `Volume` (int | float): Total units traded.
        """

        old_df = pd.read_csv(self.old_df_path)

        new_df.columns = list(map(str.capitalize, new_df.columns.to_list()))

        if sorted(new_df.columns) == sorted(old_df.columns):
            combined_df = pd.concat((old_df, new_df), ignore_index=True)
        else:
            raise ValueError(
                "Columns of new data do not match with existing dataframe")

        df = preprocess_data(combined_df)
        SFE = StockFeatureEngineer(df)
        df = SFE.create_all_features()
        X = df.drop(["Target", "Date"], axis=1)
        X = X.iloc[-(new_df.shape)[0]:]
        return X

    def load_model_pipeline(self):
        path = os.path.abspath(self.pipe_path)
        if os.path.exists(path):
            with open(path, "rb") as f:
                pipe = pickle.load(f)
            return pipe
        else:
            raise FileNotFoundError(f"Pipeline file not found at path: {self.pipe_path}")
            

