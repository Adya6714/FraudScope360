import pandas as pd
import numpy as np

def generate_data(n_users=50, n_txns=500):
    users = [f"user_{i}" for i in range(n_users)]
    txns = []

    for _ in range(n_txns):
        user = np.random.choice(users)
        amount = np.random.exponential(scale=100)  # Simulated transaction amount
        timestamp = pd.Timestamp("2025-01-01") + pd.Timedelta(seconds=int(np.random.rand() * 1e6))
        device = f"device_{np.random.randint(0, 20)}"
        ip = f"192.168.{np.random.randint(0, 255)}.{np.random.randint(0, 255)}"
        
        txns.append((user, amount, timestamp, device, ip))

    df = pd.DataFrame(txns, columns=["user", "amount", "timestamp", "device", "ip"])
    return df

def simulate_tickets(n=100):
    tickets = []
    for i in range(n):
        if i % 2 == 0:
            msg = "Transaction flagged as unusual by user"
        else:
            msg = "Unable to access account after suspected fraud"
        tickets.append(msg)
    return tickets


def simulate_edges(n_users=50, n_edges=200):
    users = [f"user_{i}" for i in range(n_users)]
    edges = []
    for _ in range(n_edges):
        u, v = np.random.choice(users, 2, replace=False)
        edges.append((u, v))
    return pd.DataFrame(edges, columns=["src", "dst"])