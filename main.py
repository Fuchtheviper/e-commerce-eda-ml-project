from ml.scripts.data_preprocessing import preprocess_data, create_sequence
from ml.scripts.train_model import train_model
from ml.scripts.predict import predict_revenue
from ml.scripts.evaluate_model import evaluate_model
from ml.scripts.hyperparameter_tuning import perform_hyperparameter_tuning, train_best_model
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    # Preprocess data
    file_path = 'data/processed_data/exported_data.csv'
    lookback = 7  # Number of time steps
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(file_path, lookback)
    X_train_seq, y_train_seq = create_sequence(X_train.values, y_train.values, lookback)
    X_val_seq, y_val_seq = create_sequence(X_val.values, y_val.values, lookback)
    X_test_seq, y_test_seq = create_sequence(X_test.values, y_test.values, lookback)
    print(f"Training shape: {X_train_seq.shape}, Validation shape: {X_val_seq.shape}, Test shape: {X_test_seq.shape}")


    # model
    model_path = 'ml/models/lstm_revenue_prediction_re2.keras'
    output_dir = 'ml/models/tuning_results'
    num_features = X_train_seq.shape[2]

    input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
    print(f"LSTM Input Shape: {input_shape}")

    tuner = perform_hyperparameter_tuning(
    X_train_seq, y_train_seq,
    X_val_seq, y_val_seq,
    lookback, num_features,
    output_dir
)

# Train the best model
    best_model = train_best_model(tuner)

    # Train model
    history = train_model(X_train_seq, y_train_seq, X_val_seq, y_val_seq, best_model)
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Predict
    y_pred = predict_revenue(best_model, X_test_seq)
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(y_test_seq)), y_test_seq, label='Actual Revenue', color='blue')
    plt.plot(range(len(y_pred)), y_pred, label='Predicted Revenue', color='red')
    plt.title('Actual vs Predicted Revenue')
    plt.xlabel('Time (Days)')
    plt.ylabel('Revenue')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Evaluate
    metrics = evaluate_model(y_test_seq, y_pred)
    print(metrics)
