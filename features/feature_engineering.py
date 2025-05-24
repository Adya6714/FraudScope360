import pandas as pd
import numpy as np

def extract_features(txns_df):

    df = txns_df.copy()  

   
    df['amount_mean'] = df.groupby('user')['amount'].transform('mean')
    df['amount_std'] = df.groupby('user')['amount'].transform('std').fillna(1)

 
    df['zscore_amount'] = (df['amount'] - df['amount_mean']) / df['amount_std']

   
    df['hour'] = df['timestamp'].dt.hour

    hours = pd.get_dummies(df['hour'], prefix='hr')

   
    df['device_count'] = df.groupby('user')['device'].transform('nunique')
    df['new_device_flag'] = np.where(df['device_count'] == 1, 1, 0)

  
    features = pd.concat([df[['zscore_amount', 'new_device_flag']], hours], axis=1)

    return features.fillna(0)