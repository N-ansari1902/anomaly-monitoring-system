import pandas as pd
import numpy as np
from utils.feature_engineering import generate_features


def generate_time_series(
    start_date="2026-01-01",
    periods=43200,
    freq="1min",
    base_value=50 
):
    """
    Generates a continuous baseline time series.

    Parameters
    ----------
    start_date : str
        Starting timestamp of the dataset.

    periods : int
        Number of rows to generate.
        Example:
        30 days of 1-minute data = 43200 rows

    freq : str
        Frequency of timestamps.

    base_value : int
        Baseline value of the signal.
    """

    timestamps = pd.date_range(start=start_date, periods=periods, freq=freq)

    # daily seasonality pattern
    daily_cycle = 10 * np.sin(np.linspace(0, 50 * np.pi, periods))

    # random noise
    noise = np.random.normal(0, 2, periods)

    values = base_value + daily_cycle + noise

    df = pd.DataFrame({
        "timestamp": timestamps,
        "value": values
    })

    return df


def inject_spike(df, index, magnitude=40):
    """Inject a spike anomaly."""
    df.loc[index, "value"] += magnitude
    return df


def inject_drop(df, index, magnitude=40):
    """Inject a drop anomaly."""
    df.loc[index, "value"] -= magnitude
    return df


def inject_level_shift(df, start_index, shift_value=15):
    """Inject a level shift anomaly."""
    df.loc[start_index:, "value"] += shift_value
    return df


def inject_drift(df, start_index, drift_rate=0.01):
    """Inject gradual drift."""
    drift = np.arange(len(df) - start_index) * drift_rate
    df.loc[start_index:, "value"] += drift
    return df


def inject_volatility(df, start_index, magnitude=10):
    """Inject volatility spike."""
    noise = np.random.normal(0, magnitude, len(df) - start_index)
    df.loc[start_index:, "value"] += noise
    return df


def create_dataset():

    df = generate_time_series()

    # Inject anomalies
    df = inject_spike(df, 5000)
    df = inject_drop(df, 10000)
    df = inject_level_shift(df, 15000)
    df = inject_drift(df, 25000)
    df = inject_volatility(df, 35000)

    return df


if __name__ == "__main__":

    # Generate raw signal
    df = create_dataset()

    # Generate monitoring features
    df = generate_features(df)

    # Save processed dataset
    df.to_csv("data/processed_anomaly_data.csv", index=False)

    print("Processed dataset generated successfully")