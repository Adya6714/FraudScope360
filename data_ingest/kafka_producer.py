import json
import time
import random
import numpy as np
import pandas as pd
from confluent_kafka import Producer

p = Producer({'bootstrap.servers': 'localhost:9092'})

def generate_txn():
    user = f"user_{random.randint(0, 49)}"
    amount = float(np.random.exponential(scale=100))
    timestamp = pd.Timestamp.now().isoformat()
    device = f"device_{random.randint(0, 20)}"
    ip = f"192.168.{random.randint(0, 255)}.{random.randint(0, 255)}"
    return {
        "user": user,
        "amount": amount,
        "timestamp": timestamp,
        "device": device,
        "ip": ip
    }

while True:
    txn = generate_txn()
    p.produce('txns', key=txn['user'], value=json.dumps(txn))
    p.flush()
    print("Produced:", txn)
    time.sleep(1)