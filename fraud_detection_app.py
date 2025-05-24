import pandas as pd
from features.feature_engineering import extract_features

# Simulate a tiny transaction dataframe
data = [
    {"user": "user_1", "amount": 120.0, "timestamp": pd.Timestamp("2025-05-24 01:00:00"), "device": "device_1", "ip": "192.168.1.1"},
    {"user": "user_1", "amount": 80.0,  "timestamp": pd.Timestamp("2025-05-24 03:00:00"), "device": "device_1", "ip": "192.168.1.1"},
    {"user": "user_2", "amount": 300.0, "timestamp": pd.Timestamp("2025-05-24 20:00:00"), "device": "device_2", "ip": "192.168.1.2"}
]
df = pd.DataFrame(data)

# Extract features
features = extract_features(df)

print(features)
