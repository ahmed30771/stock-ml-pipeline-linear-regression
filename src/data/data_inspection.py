import pandas as pd
from ..data import load_config


class DataInspector:
    def __init__(self):
        self.config = load_config()

    def data_types(self, df: pd.DataFrame):
        dtyp = df.dtypes.to_frame(name="Dtype")
        return dtyp

    def missing_values(self, df: pd.DataFrame):
        missing = df.isnull().sum(axis=0).to_frame(name="Missing values")
        return missing


def data_inspection_report(df: pd.DataFrame, Title="Data Inspection Report"):
    """
    Inspect stock data from a CSV file and print.

    Args
    -------
    df: pd.DataFrame
        DataFrame with stock data

    Returns
    -------
        str
    """
    DI = DataInspector()
    data_types = DI.data_types(df)
    missing = DI.missing_values(df)
    result = pd.concat((data_types, missing), axis="columns")
    return f'''{'='*100}
{' '*30} {Title}
{'='*100}
→ Number of Rows = {df.shape[1]}
→ Number of Columns = {df.shape[1]}
→ Duplicated Rows = {int(df.T.duplicated().sum())}
{'-'*70}
{result}
'''
