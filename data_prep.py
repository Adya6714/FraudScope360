# data_prep.py

import pandas as pd
import numpy as np

print("👉 data_prep.py has started")

# 1️⃣ Read only the columns we need from train_transaction
tx_cols = [
    "TransactionID","TransactionDT","TransactionAmt",
    "ProductCD","card1","card4","addr1","R_emaildomain","isFraud"
]
#step 1 Read only the columns we need from train_transaction
print("Reading train_transaction.csv (selected columns)…")
tx = pd.read_csv(
    "data/train_transaction.csv",
    usecols=tx_cols,
    low_memory=False
)
# ensure TransactionDT is numeric
tx["TransactionDT"] = pd.to_numeric(tx["TransactionDT"], errors="coerce")

print(f" ↳ Loaded tx shape: {tx.shape}")

# 2 Read only TransactionID from train_identity
print("Reading train_identity.csv (TransactionID + nothing else)…")
idf = pd.read_csv(
    "data/train_identity.csv",
    usecols=["TransactionID"], 
    low_memory=False
)
print(f" ↳ Loaded idf shape: {idf.shape}")

# 3 Merge them on TransactionID
print("Merging transaction + identity…")
df = tx.merge(idf, on="TransactionID", how="left")
print(f" ↳ Merged shape: {df.shape}")

# 4 Create real timestamp
print("Converting TransactionDT (seconds) → real datetime…")
origin = pd.Timestamp("2017-11-30")
df["timestamp"] = origin + pd.to_timedelta(df["TransactionDT"].astype(float), unit="s")
print(" ↳ timestamp dtype is now", df["timestamp"].dtype)

# 5 Select & rename into our eight-column schema
print("Selecting & renaming columns…")
out = df[[
    "card1", "TransactionAmt", "timestamp",
    "card4", "R_emaildomain",
    "ProductCD", "addr1", "isFraud"
]].rename(columns={
    "card1":          "user",
    "TransactionAmt": "amount",
    "card4":          "device",
    "R_emaildomain":  "ip",
    "ProductCD":      "merchant",
    "addr1":          "country",
    "isFraud":        "is_fraud"
})
# Make sure user IDs are strings everywhere
out["user"] = out["user"].astype(str)

print(f" ↳ After rename: {out.shape}")

# 6 Drop any rows missing essential fields
print("Dropping rows missing core fields…")
before = len(out)
out = out.dropna(subset=[
    "user","amount","timestamp","device","ip","merchant","country","is_fraud"
])
print(f" ↳ Dropped {before - len(out)} rows; remaining {len(out)} rows")

# 7 Write out the final CSV
print("Writing data/transactions_labeled.csv…")
out.to_csv("data/transactions_labeled.csv", index=False)
print(" Done! Wrote data/transactions_labeled.csv →", out.shape)