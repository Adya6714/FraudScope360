import json
import pandas as pd
from datetime import datetime
from confluent_kafka import Consumer

from features.feature_engineering import extract_features
from modules.anomaly.isolation_forest import AnomalyDetector
# import other modules like changepoint, graph, identity, NLP

# Setup Kafka consumer
consumer = Consumer({
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'fraud-detector',
    'auto.offset.reset': 'latest'
})

consumer.subscribe(['txns'])

# Load pre-trained module (static training)
history = pd.DataFrame([  # simulate history
    {"user": f"user_{i}", "amount": float(np.random.exponential(100)), 
     "timestamp": pd.Timestamp("2025-01-01"), 
     "device": f"device_{i%10}", "ip": f"192.168.1.{i%255}"}
    for i in range(100)
])
features = extract_features(history)
anomaly = AnomalyDetector(); anomaly.fit(features)

print("🚀 Real-time Fraud Detection Started")
while True:
    msg = consumer.poll(1.0)
    if msg is None:
        continue
    if msg.error():
        print("Error:", msg.error())
        continue

    txn = json.loads(msg.value().decode('utf-8'))

    # Convert to DataFrame
    df = pd.DataFrame([txn])
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Extract features
    feat = extract_features(df)

    # Score with modules
    score = anomaly.predict(feat)[0]
    print(f"Scored Transaction: {txn} -> Anomaly Score: {score:.3f}")