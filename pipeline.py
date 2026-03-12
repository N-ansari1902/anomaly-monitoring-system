import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from utils.feature_engineering import generate_features


# -------------------------------------------------
# Load model and scaler once
# -------------------------------------------------

model = load_model("models/lstm_anomaly_model.h5")
scaler = joblib.load("models/scaler.pkl")


# -------------------------------------------------
# Function to create LSTM sequences
# -------------------------------------------------

def create_sequences(data, window_size=50):

    sequences = []

    for i in range(len(data) - window_size):
        seq = data[i:i + window_size]
        sequences.append(seq)

    return np.array(sequences)


# -------------------------------------------------
# Main pipeline function
# -------------------------------------------------

def run_pipeline(file_path):

    print(f"Processing file: {file_path}")

    # ---------------------------------------------
    # Load dataset
    # ---------------------------------------------

    df = pd.read_csv(file_path)

    # Convert timestamp column
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # ---------------------------------------------
    # Feature Engineering
    # ---------------------------------------------

    df = generate_features(df)

    # rename columns to match training
    df["rolling_max"] = df["rolling_max_5"]
    df["rolling_min"] = df["rolling_min_5"]

    # ---------------------------------------------
    # Feature selection for model
    # (DO NOT CHANGE ORDER)
    # ---------------------------------------------

    features = [
        "value",
        "rolling_mean_5",
        "rolling_std_5",
        "value_diff",
        "pct_change",
        "z_score",
        "rolling_max",
        "rolling_min"
    ]

    X = df[features].astype(float)

    # ---------------------------------------------
    # Scaling
    # ---------------------------------------------

    X_scaled = scaler.transform(X)

    # ---------------------------------------------
    # Create LSTM sequences
    # ---------------------------------------------

    window_size = 50

    X_seq = create_sequences(X_scaled, window_size)

    # ---------------------------------------------
    # Model prediction
    # ---------------------------------------------

    predictions = model.predict(X_seq)

    # flatten predictions if needed
    predictions = predictions.flatten()

    # ---------------------------------------------
    # Align dataframe with sequence output
    # ---------------------------------------------

    df = df.iloc[window_size:].copy()

    # add predictions
    df["anomaly_score"] = predictions

    # ---------------------------------------------
    # Threshold based anomaly detection
    # ---------------------------------------------

    df["model_anomaly"] = (df["anomaly_score"] > 0.8).astype(int)

    return df