# features/feature_engineering.py

import pandas as pd
import numpy as np

def extract_features(txns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with columns:
      user, amount, timestamp, device, ip, merchant, country
    returns a feature‐matrix DataFrame with:
      - zscore_amount
      - new_device_flag
      - new_ip_flag
      - new_merchant_flag
      - new_country_flag
      - hr_0 … hr_23
    """
    df = txns_df.copy()
    #  ensure timestamp column is real datetime
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
       df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 1️⃣ Standardize amount per user
    df["amount_mean"] = df.groupby("user")["amount"].transform("mean")
    df["amount_std"]  = df.groupby("user")["amount"].transform("std").fillna(1.0)
    df["zscore_amount"] = (df["amount"] - df["amount_mean"]) / df["amount_std"]

    # 2️⃣ Time of day – one‐hot encode hour
    df["hour"] = df["timestamp"].dt.hour
    hr_dummies = pd.get_dummies(df["hour"], prefix="hr")

    # 3️⃣ New‐entity flags: device, ip, merchant, country
    for col in ["device", "ip", "merchant", "country"]:
        count_col = f"{col}_count"
        flag_col  = f"new_{col}_flag"
        df[count_col] = df.groupby("user")[col].transform("nunique")
        df[flag_col]  = np.where(df[count_col] == 1, 1, 0)

    # 4️⃣ Assemble feature matrix
    features = pd.concat(
        [
            df[["zscore_amount",
                "new_device_flag",
                "new_ip_flag",
                "new_merchant_flag",
                "new_country_flag"]],
            hr_dummies
        ],
        axis=1
    )

    # 5️⃣ Fill any NaNs with zero
    return features.fillna(0)