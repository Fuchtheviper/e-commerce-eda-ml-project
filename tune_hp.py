from ml.scripts.data_preprocessing import preprocess_data, create_sequence
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
import pandas as pd

file_path = 'data/processed_data/exported_data.csv'
model_path = 'ml/models/lstm_revenue_prediction_re2.keras'
lookback = 7  # Number of time steps
X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(file_path, lookback)
X_train_seq, y_train_seq = create_sequence(X_train.values, y_train.values, lookback)
X_val_seq, y_val_seq = create_sequence(X_val.values, y_val.values, lookback)
X_test_seq, y_test_seq = create_sequence(X_test.values, y_test.values, lookback)
input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
lookback = X_train_seq.shape[1]
num_features = X_train_seq.shape[2]

# Perform hyperparameter tuning
output_dir = 'ml/models/tuning_results'
#tuner = perform_hyperparameter_tuning(
    #X_train_seq, y_train_seq,
    #X_val_seq, y_val_seq,
    #lookback=lookback, 
    #num_features=num_features,
    #output_dir=output_dir
#)

# Retrieve the best model
#best_model = get_best_model(tuner)
#best_model.summary()

def build_model(hp):
    model = Sequential()
    # Tune number of units in the first LSTM layer
    model.add(LSTM(
        units=hp.Int('units', min_value=32, max_value=128, step=32),
        activation='relu',
        input_shape=(lookback, num_features),
        return_sequences=True
    ))
    model.add(Dropout(hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)))
    # Tune the number of layers
    model.add(LSTM(
        units=hp.Int('units_layer2', min_value=32, max_value=128, step=32),
        activation='relu'
    ))
    model.add(Dense(1))
    # Tune the learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Choice('learning_rate', values=[0.001, 0.01, 0.0001])
        ),
        loss='mse'
    )
    return model

tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=2,
    directory=output_dir,
    project_name='lstm_tuning'
)

tuner.search(X_train_seq, y_train_seq, epochs=15, validation_data=(X_val_seq, y_val_seq), batch_size=16)
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The optimal number of units in the first LSTM layer is {best_hps.get('units')}.
The optimal dropout rate is {best_hps.get('dropout')}.
The optimal number of units in the second LSTM layer is {best_hps.get('units_layer2')}.
The optimal learning rate is {best_hps.get('learning_rate')}.
""")
best_model = tuner.hypermodel.build(best_hps)

best_model.save(model_path)