import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(file_path,lookback):
    raw_data = pd.read_csv(file_path)
    # Define columns
    numeric_columns = raw_data.select_dtypes(include=[np.number]).columns
    category_columns = raw_data.select_dtypes(include=[object]).columns
    selected_columns = ['Revenue', 'Units_Sold', 'Discount_Amount', 'Ad_Spend', 'Clicks', 'Impressions']
    selected_daily_num_columns = ['Revenue_Daily_Sum', 'Units_Sold_Daily_Sum',
        'Discount_Amount_Daily_Sum', 'Ad_Spend_Daily_Sum', 'Clicks_Daily_Sum',
        'Impressions_Daily_Sum']
    target_column = 'Revenue_Daily_Sum'

    missing_handled_data = handle_missing_values(raw_data, numeric_columns, category_columns)
    infinite_handled_data = handle_infinite_value(missing_handled_data, numeric_columns)
    daily_sum_data = aggregate_data_to_daily(infinite_handled_data, selected_columns)
    daily_featured_data = add_time_based_features(daily_sum_data)
    outlier_removed_data = handle_extreme_outlier(daily_featured_data, selected_daily_num_columns)
    normalized_data = normalize_data(outlier_removed_data, selected_daily_num_columns)
    encoded_data = encode_data(normalized_data)
    datatype_converted_data = convert_data_type(encoded_data)
    lagged_feature_data = create_lagged_features(datatype_converted_data, selected_daily_num_columns, lookback)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(lagged_feature_data, target_column)
    return X_train, X_val, X_test, y_train, y_val, y_test

def handle_missing_values(data, numeric_columns, category_columns):
    # Fill missing numerical values with the column mean
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

    # Fill missing categorical values with the mode
    data[category_columns] = data[category_columns].fillna(data[category_columns].mode().iloc[0])

    # Verify if null values are handled
    print("Null Values After Handling:\n", data.isnull().sum())
    return data

def handle_infinite_value(data, numeric_columns):
    # Replace positive and negative infinity with NaN
    data[numeric_columns] = data[numeric_columns].replace([np.inf, -np.inf], np.nan)

    # Fill NaN (resulting from inf) with the mean of each column
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

    # Verify the dataset
    print("Infinity Values After Handling:\n", np.isinf(data[numeric_columns]).sum())
    print("Null Values After Handling:\n", data.isnull().sum())
    return data

def aggregate_data_to_daily(data, selected_columns):
    # Ensure 'Transaction_Date' is in datetime format
    data['Transaction_Date'] = pd.to_datetime(data['Transaction_Date'])

    aggregated_data = {}

    # Loop through each selected column and aggregate
    for column in selected_columns:
        aggregated_column = data.groupby('Transaction_Date')[column].sum().reset_index()
        aggregated_column.rename(columns={column: f'{column}_Daily_Sum'}, inplace=True)
        aggregated_data[column] = aggregated_column

    # Merge aggregated columns into a single DataFrame
    daily_data = aggregated_data[selected_columns[0]]  # Start with the first column
    for column in selected_columns[1:]:
        daily_data = daily_data.merge(aggregated_data[column], on='Transaction_Date', how='left')

    print("Data amount after aggergated:\n", daily_data.count())
    return daily_data

def add_time_based_features(daily_data):
    # Add time-based features directly from Transaction_Date
    daily_data['Day_of_Week'] = daily_data['Transaction_Date'].dt.dayofweek + 1  # Monday=1, Sunday=7
    daily_data['Is_Weekend'] = daily_data['Day_of_Week'].apply(lambda x: 1 if x in [6, 7] else 0)
    daily_data['Year'] = daily_data['Transaction_Date'].dt.year
    daily_data['Month'] = daily_data['Transaction_Date'].dt.month
    print("Data amount after time feature added:\n", daily_data.count())
    return daily_data

def handle_extreme_outlier(daily_data, selected_daily_num_columns):
    for col in selected_daily_num_columns:
        lower_limit = daily_data[col].quantile(0.01)
        upper_limit = daily_data[col].quantile(0.99)
        daily_data[col] = np.clip(daily_data[col], lower_limit, upper_limit)
        print("Data amount after remove extreme outlier:\n", daily_data.count())
    return daily_data

def normalize_data(daily_data, selected_daily_num_columns):
    scaler = StandardScaler()
    daily_data[selected_daily_num_columns] = scaler.fit_transform(daily_data[selected_daily_num_columns])
    print("Data amount after remove extreme outlier:\n", daily_data.head())
    return daily_data

def encode_data(daily_data):
    daily_data = pd.get_dummies(daily_data, columns=['Day_of_Week', 'Is_Weekend','Year','Month'], drop_first=True)
    print("Data amount after encoded:\n", daily_data.count())
    print("Data type:\n", daily_data.select_dtypes(include=['object']).columns)
    return daily_data

def convert_data_type(daily_data):
    # Convert boolean columns to integers
    daily_data = daily_data.astype({col: 'int' for col in daily_data.select_dtypes(include=['bool']).columns})
    print("Data type:\n", daily_data.select_dtypes(include=['object']).columns)
    return daily_data

def create_lagged_features(daily_data, selected_daily_num_columns, lookback):
    # Create lagged features
    for col in selected_daily_num_columns:
        for lag in range(1, lookback + 1):
            daily_data[f'{col}_lag_{lag}'] = daily_data[col].shift(lag)

    # Drop rows with NaN values caused by lagging
    lookback_daily_data = daily_data.dropna().reset_index(drop=True)
    return lookback_daily_data

def split_data(lookback_daily_data, target_column):
    # Define target and features
    X = lookback_daily_data.drop(columns=['Transaction_Date',target_column])
    y = lookback_daily_data[target_column]

    # Split into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)
    print(f"Training shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_sequence(feature, target, lookback):
    X, y = [], []
    for i in range(len(feature) - lookback):
        X.append(feature[i:i+lookback])
        y.append(target[i+lookback])
    return np.array(X), np.array(y)