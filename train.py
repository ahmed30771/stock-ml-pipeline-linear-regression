import pandas as pd
from src import load_config, load_data, data_inspection_report, preprocess_data, StockFeatureEngineer, StockPriceModel, ModelEvaluator


def train_model():
    """
    This function loads the data, processes it, and runs the machine learning model.

    Returns
    -------
        str
    """

    f = open("Report.txt", "w+", encoding="UTF-8")

    # ------------------------------------------------------
    # Configuration
    # ------------------------------------------------------
    config = load_config()

    # ------------------------------------------------------
    # Data
    # ------------------------------------------------------

    # load data
    df = load_data()

    # data inspection
    print(data_inspection_report(df, Title="Data Inspection Report"), file=f)

    # data preprocessing
    df = preprocess_data(df)

    # ------------------------------------------------------
    # Features
    # ------------------------------------------------------
    SFE = StockFeatureEngineer(df)
    df = SFE.create_all_features()

    # data inspection of processed data
    print(data_inspection_report(
        df, Title="Data Inspection Report Of Processed Data"), file=f)

    print("-"*100, file=f)
    print(SFE.save_processed_data(), file=f)

    # ------------------------------------------------------
    # Train Model Pipeline
    # ------------------------------------------------------
    SPM = StockPriceModel(df)

    # train test split
    X_train, X_test, y_train, y_test = SPM.split_data()

    # load model pipeline
    pipe = SPM.model_pipeline(X_train, y_train)

    # Save pipeline
    print(SPM.save_pipeline(pipe), file=f)
    print("-"*100, end="\n\n", file=f)

    # ------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------
    ME = ModelEvaluator(model_pipe=pipe)

    # evaluate and print results
    print("="*100, " " * 31 + "Model Evaluation Metrics",
          "="*100, sep="\n", file=f)
    print(ME.evaluate_train_test(X_train, X_test, y_train, y_test), file=f)

    f.seek(0)
    report = f.read()
    f.close()
    return report


if __name__ == "__main__":
    print(train_model())
