import pandas as pd
import numpy as np


def generate_features(df):
    """
    Takes raw signal data and generates monitoring features.
    """

    df = df.copy()

    # --------------------------------------------
    # Rolling statistics
    # --------------------------------------------

    df["rolling_mean_5"] = df["value"].rolling(window=5).mean()
    df["rolling_std_5"] = df["value"].rolling(window=5).std()

    df["rolling_max_5"] = df["value"].rolling(window=5).max()
    df["rolling_min_5"] = df["value"].rolling(window=5).min()

    # --------------------------------------------
    # Value difference
    # --------------------------------------------

    df["value_diff"] = df["value"].diff()

    # --------------------------------------------
    # Percent change
    # --------------------------------------------

    df["pct_change"] = df["value"].pct_change()

    # --------------------------------------------
    # Z-score calculation
    # --------------------------------------------

    mean_val = df["value"].mean()
    std_val = df["value"].std()

    df["z_score"] = (df["value"] - mean_val) / std_val

    # --------------------------------------------
    # Anomaly rule
    # --------------------------------------------

    df["anomaly"] = 0
    df.loc[df["z_score"].abs() > 3, "anomaly"] = 1

    # --------------------------------------------
    # Anomaly type classification
    # --------------------------------------------

    df["anomaly_type"] = "normal"

    df.loc[(df["anomaly"] == 1) & (df["value_diff"] > 20), "anomaly_type"] = "spike"
    df.loc[(df["anomaly"] == 1) & (df["value_diff"] < -20), "anomaly_type"] = "drop"

    df.loc[(df["anomaly"] == 1) & (df["rolling_std_5"] > 10), "anomaly_type"] = "volatility"

    # --------------------------------------------
    # Fill missing values
    # --------------------------------------------

    df = df.fillna(0)

    return df