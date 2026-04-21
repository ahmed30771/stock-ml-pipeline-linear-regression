import pandas as pd
import yaml
import os


def load_config(config_path="./config/config.yaml"):
    """
    Load configuration from YAML file.

    Args
    -------
    config_path: str.
        Path to the configuration.<br>
        Defaults to "./config/config.yaml"

    Returns
    -------
    dict
        Dictionary with configuration settings
    """
    config_path = os.path.abspath(config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_data():
    """
    Load stock data from a CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame of stock data
    """
    file_path = os.path.join(os.path.abspath(
        load_config()['data']['file_path']), load_config()['symbol'] + ".csv")
    df = pd.read_csv(file_path, index_col=False)
    return df


