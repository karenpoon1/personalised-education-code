import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# extract properties of columns from .csv

df = pd.read_csv("personalised-education/Fwd__Pinpoint_ML_Dataset/9to1_2017_GCSE_1H.csv", skiprows = [i for i in range(1,24)], usecols=[i for i in range(2,26)])

# binarise data using average
for col in df:
    mean_score = df[col].mean()
    df[col] = (df[col] >= mean_score).astype(float)

# train, test = train_test_split(df, test_size=0.9, random_state=42, shuffle=True)
# shuffled_df = df.reindex(np.random.permutation(df.index))
    # shuffle rows
    # student_baseline_df = student_baseline_df.sample(frac=1, axis=0, random_state=random_seed)
    # reset row index (from student 0 to end, columns remain shuffled)
    # student_baseline_df = student_baseline_df.reset_index(drop=True)

def train_student_baseline(random_seed, validation_split, df, no_samples, metric):

    student_baseline_df = df.head(no_samples)
    no_questions = len(student_baseline_df.columns)

    # shuffle cols
    student_baseline_df = student_baseline_df.sample(frac=1, axis=1, random_state=random_seed)

    # split to train and test set
    no_train_cols = int(no_questions * validation_split)
    train_df = student_baseline_df.iloc[:, :no_train_cols]
    test_df = student_baseline_df.iloc[:, no_train_cols:]

    # compute probit of a student answering a question correctly, for each student
    student_probit = train_df.sum(axis=1)/no_train_cols

    predictions_df = pd.DataFrame().reindex_like(test_df)

    no_test_cols = no_questions - no_train_cols
    for i in range(len(student_probit)):
        predictions_df.iloc[i] = np.random.binomial(size=no_test_cols, n=1, p=student_probit[i])

    total_entries = test_df.shape[0]*test_df.shape[1]

    if metric == 'percentage correct':
        no_correct_predictions = np.count_nonzero(predictions_df == test_df)
        performance = no_correct_predictions/total_entries
    elif metric == 'cf':
        true_pos, false_pos, true_neg, false_neg = 0,0,0,0
        for col in predictions_df:
            temp_cf = pd.crosstab(predictions_df[col], test_df[col], rownames=['Actual'], colnames=['Predicted'])
            true_pos += temp_cf.iloc[1,1]
            false_pos += temp_cf.iloc[0,1]
            true_neg += temp_cf.iloc[0,0]
            false_neg += temp_cf.iloc[1,0]
        performance = f"true_pos: {true_pos/total_entries}, false_pos: {false_pos/total_entries}, true_neg: {true_neg/total_entries}, false_neg: {false_neg/total_entries}"
    return performance

def train_student_shuffle_each_row(validation_split, df, no_samples):

    student_baseline_df = df.head(no_samples)
    no_questions = len(student_baseline_df.columns)

    # split to train and test set
    no_train_cols = int(no_questions * validation_split)
    no_test_cols = no_questions - no_train_cols
    no_correct_predictions = 0

    # shuffle cols
    for _, row in student_baseline_df.iterrows():
        shuffled_row = row.sample(frac=1)
        train_row = shuffled_row.iloc[:no_train_cols]
        test_row = shuffled_row.iloc[no_train_cols:]
        test_df = test_row.to_frame()
        student_probit = train_row.sum()/no_train_cols
        prediction_df = pd.DataFrame().reindex_like(test_df)
        prediction_df.iloc[:, 0] = np.random.binomial(size=no_test_cols, n=1, p=student_probit)
        no_correct_predictions += np.count_nonzero(prediction_df == test_df)

    total_entries = no_samples * no_test_cols
    performance = no_correct_predictions/total_entries
    
    return performance


for i in range(1):
    print(train_student_baseline(20, 0.5, df, len(df), 'cf'))

# print(train_student_shuffle_each_row(0.5, df, len(df)))