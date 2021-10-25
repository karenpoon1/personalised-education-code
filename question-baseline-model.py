import numpy as np
from numpy.random import default_rng
import pandas as pd
from sklearn.model_selection import train_test_split

# extract properties of columns from .csv

df = pd.read_csv("personalised-education/Fwd__Pinpoint_ML_Dataset/9to1_2017_GCSE_1H.csv", skiprows = [i for i in range(1,24)], usecols=[i for i in range(2,26)])

for col in df:
    mean_score = df[col].mean()
    df[col] = (df[col] >= mean_score).astype(float) # binarise data

def train_question_baseline(seed_number, validation_split, df, no_samples):
    
    rng = default_rng(seed=seed_number)
    student_baseline_df = df.head(no_samples)

    # shuffle rows
    # student_baseline_df = student_baseline_df.sample(frac=1, axis=0, random_state=np.random.RandomState(seed_number))
    # student_baseline_df = student_baseline_df.sample(frac=1, axis=1, random_state=np.random.RandomState(seed_number))
    student_baseline_df = student_baseline_df.reset_index(drop=True)

    # split to train and test set
    no_train_rows = int(no_samples * validation_split)
    train_df = student_baseline_df.iloc[:no_train_rows, 12:]
    test_df = student_baseline_df.iloc[no_train_rows:, 12:]

    # compute probit of a question being answered correctly, for each question
    question_probit = train_df.sum(axis=0)/no_train_rows
    print(question_probit)
    
    predictions_df = pd.DataFrame().reindex_like(test_df)

    no_test_rows = no_samples - no_train_rows
    for i in range(len(question_probit)):
        predictions_df.iloc[:, i] = rng.binomial(size=no_test_rows, n=1, p=question_probit[i])

    no_correct_predictions = np.count_nonzero(predictions_df == test_df)
    total_entries = test_df.shape[0]*test_df.shape[1]
    performance = no_correct_predictions/total_entries

    print(performance)
    return performance

performance_arr = [train_question_baseline(i, 0.5, df, len(df))*100 for i in range(10)]
print(performance_arr)
print(np.mean(performance_arr), np.std(performance_arr))