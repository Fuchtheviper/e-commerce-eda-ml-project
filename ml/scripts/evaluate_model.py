from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def evaluate_model(y_true, y_pred):
    """
    Evaluates the performance of a regression model using Mean Absolute Error (MAE) and
    Root Mean Squared Error (RMSE).

    Params:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        dict: A dictionary containing the calculated MAE and RMSE metrics.
        - "MAE": Mean Absolute Error.
        - "RMSE": Root Mean Squared Error.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse}