import tensorflow as tf
import numpy as np

def predict_revenue(model_path, X_test):
    model = tf.keras.models.load_model(model_path)
    predictions = model.predict(X_test)
    return predictions