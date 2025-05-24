import pandas as pd

def read_csv_data(filepath="data/transactions.csv"):
    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    return df


from sqlalchemy import create_engine
import pandas as pd

def read_db_data():
    engine = create_engine('sqlite:///data/txns.db')
    df = pd.read_sql("SELECT * FROM transactions", engine)
    return df