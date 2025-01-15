import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib

def train_model(X_train, y_train, X_val, y_val, model_path):
    model = tf.keras.models.load_model(model_path)
    #early_stopping = EarlyStopping(monitor='val_loss', patience=500, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=500,
        batch_size=16,
        verbose=1
    )
    return history