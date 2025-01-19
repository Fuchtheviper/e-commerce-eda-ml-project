import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib

def train_model(X_train, y_train, X_val, y_val, best_model):
    """
    Trains the given model using the provided training and validation datasets.

    Params:
        X_train (np.ndarray): Training feature data.
        y_train (np.ndarray): Training target data.
        X_val (np.ndarray): Validation feature data.
        y_val (np.ndarray): Validation target data.
        best_model (tf.keras.Model): The pre-built and compiled Keras model to train.

    Returns:
        history (tf.keras.callbacks.History): Training history object containing details about the training process,
        such as loss and validation loss per epoch.
    """
    history = best_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=500,
        batch_size=16,
        verbose=1
    )
    return history