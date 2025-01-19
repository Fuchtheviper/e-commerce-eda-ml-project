from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
import keras_tuner as kt

def build_model(hp, lookback, num_features):
    """
    Build an LSTM model with hyperparameter tuning.

    Params:
        hp: Hyperparameter object from Keras Tuner.
        lookback (int): Number of time steps for input sequences.
        num_features (int): Number of features in the input data.

    Returns:
        Compiled Keras model.
    """
    model = Sequential()
    # First LSTM layer
    model.add(LSTM(
        units=hp.Int('units', min_value=32, max_value=128, step=32),
        activation='relu',
        input_shape=(lookback, num_features),
        return_sequences=True
    ))
    model.add(Dropout(hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)))

    # Second LSTM layer
    model.add(LSTM(
        units=hp.Int('units_layer2', min_value=32, max_value=128, step=32),
        activation='relu'
    ))

    # Output layer
    model.add(Dense(1))

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Choice('learning_rate', values=[0.001, 0.01, 0.0001])
        ),
        loss='mse'
    )
    return model

def perform_hyperparameter_tuning(X_train_seq, y_train_seq, X_val_seq, y_val_seq, lookback, num_features, output_dir):
    """
    Perform hyperparameter tuning using Keras Tuner.

    Params:
        X_train_seq (np.ndarray): Training sequences.
        y_train_seq (np.ndarray): Training targets.
        X_val_seq (np.ndarray): Validation sequences.
        y_val_seq (np.ndarray): Validation targets.
        lookback (int): Number of time steps for input sequences.
        num_features (int): Number of features in the input data.
        output_dir (str): Directory to save tuning results.

    Returns:
        kt.Hyperband: Tuner object after hyperparameter tuning.
    """
    # Set up the tuner
    tuner = kt.RandomSearch(
        lambda hp: build_model(hp, lookback, num_features),
        objective='val_loss',
        max_trials=10,
        executions_per_trial=2,
        directory=output_dir,
        project_name='lstm_tuning'
    )

    # Perform tuning
    tuner.search(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=15,
        batch_size=16
    )

    return tuner

def perform_hyperparameter_tuning(X_train_seq, y_train_seq, X_val_seq, y_val_seq, lookback, num_features, output_dir):
    """
    Perform hyperparameter tuning using Keras Tuner.

    Params:
        X_train_seq (np.ndarray): Training sequences.
        y_train_seq (np.ndarray): Training targets.
        X_val_seq (np.ndarray): Validation sequences.
        y_val_seq (np.ndarray): Validation targets.
        lookback (int): Number of time steps for input sequences.
        num_features (int): Number of features in the input data.
        output_dir (str): Directory to save tuning results.

    Returns:
        kt.Hyperband: Tuner object after hyperparameter tuning.
    """
    # Set up the tuner
    tuner = kt.RandomSearch(
        lambda hp: build_model(hp, lookback, num_features),
        objective='val_loss',
        max_trials=10,
        executions_per_trial=2,
        directory=output_dir,
        project_name='lstm_tuning'
    )

    # Perform tuning
    tuner.search(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=15,
        batch_size=16
    )

    return tuner

def train_best_model(tuner):
    """
    Train the best model using the hyperparameters obtained from the tuner.

    Params:
        tuner (kt.Hyperband): Tuner object containing hyperparameter results.
        X_train_seq (np.ndarray): Training sequences.
        y_train_seq (np.ndarray): Training targets.
        X_val_seq (np.ndarray): Validation sequences.
        y_val_seq (np.ndarray): Validation targets.
        model_path (str): Path to save the best model.

    Returns:
        history: Training history of the model.
        best_model: Trained Keras model.
    """
    # Retrieve the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"""
    The optimal number of units in the first LSTM layer is {best_hps.get('units')}.
    The optimal dropout rate is {best_hps.get('dropout')}.
    The optimal number of units in the second LSTM layer is {best_hps.get('units_layer2')}.
    The optimal learning rate is {best_hps.get('learning_rate')}.
    """)

    # Build the best model
    best_model = tuner.hypermodel.build(best_hps)
    return best_model
