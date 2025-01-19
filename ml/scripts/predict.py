import tensorflow as tf
import numpy as np

def predict_revenue(best_model, X_test):
    predictions = best_model.predict(X_test)
    return predictions