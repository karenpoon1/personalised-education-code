def binarise_by_avg(df):
    """
    Binarise data using mean score over students for each question
        - df: pd.DataFrame() to be binarised
    """
    for col in df:
        mean_score = df[col].mean()
        df[col] = (df[col] >= mean_score).astype(float)
    return df

def binarise_by_mid(df, max_scores):
    """
    Binarise data using mid value of max score for each question
        - df: pd.DataFrame() to be binarised
        - max_scores: pd.DataFrame() containing max score for each question
    """
    for col in df:
        max_score = max_scores[col].iloc[0]
        df[col] = (df[col] >= max_score/2).astype(float)
    return df
