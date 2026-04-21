from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import numpy as np
import pandas as pd


class ModelEvaluator:
    def __init__(self, model_pipe):
        self.model_pipe = model_pipe

    def predict(self, X):
        pred = self.model_pipe.predict(X)
        return pred

    def calculate_metrics(self, X, y):
        pred = self.predict(X)
        mae = mean_absolute_error(y, pred)
        mse = mean_squared_error(y, pred)
        rmse = round(np.sqrt(mse).item(), 4)
        mape = mean_absolute_percentage_error(y, pred)
        r2 = r2_score(y, pred)

        actual_direction = np.sign(np.diff(y))
        predicted_direction = np.sign(np.diff(pred))
        directional_accuracy = round(
            np.mean(actual_direction == predicted_direction).item(), 4)
        metrics = {
            "MAE": [mae],
            "MSE": [mse],
            "RMSE": [rmse],
            "MAPE": [mape],
            "R2 Score": [r2],
            'Directional Accuracy': [directional_accuracy]
        }
        return metrics

    def evaluate_train_test(self, X_train, X_test, y_train, y_test):
        eval_train = pd.DataFrame(self.calculate_metrics(X=X_train, y=y_train), index=["Train"])
        eval_test = pd.DataFrame(self.calculate_metrics(X=X_test, y=y_test), index=["Test"])
        concat_evals = pd.concat((eval_train, eval_test), axis='index')
        return concat_evals

