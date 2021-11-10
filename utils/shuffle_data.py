import numpy as np

def shuffle_rows(df, shuffle_seed=1000): # no impact
    df = df.sample(frac=1, axis=0, random_state=np.random.RandomState(shuffle_seed))
    df = df.reset_index(drop=True)
    return df

def shuffle_cols(df, shuffle_seed=1000):
    df = df.sample(frac=1, axis=1, random_state=np.random.RandomState(shuffle_seed))
    return df
