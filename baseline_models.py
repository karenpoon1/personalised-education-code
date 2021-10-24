import numpy as np
from numpy.random import default_rng
import pandas as pd

df = pd.read_csv("personalised-education/Fwd__Pinpoint_ML_Dataset/9to1_2017_GCSE_1H.csv", skiprows=list(range(1,24)), usecols=list(range(2,26)))

# binarise data using average
for col in df:
    mean_score = df[col].mean()
    df[col] = (df[col] >= mean_score).astype(float)

# split to train and test set
student_split = 0.5
question_split = 0.5

no_train_rows = int(len(df) * student_split)
no_train_cols = int(len(df.columns) * question_split)

train_question_df = df.iloc[:no_train_rows, no_train_cols:]
train_question_df = train_question_df.reset_index(drop=True)
train_student_df = df.iloc[no_train_rows:, :no_train_cols]
test_df = df.iloc[no_train_rows:, no_train_cols:]

def train_student_baseline(train_df, test_df, seed_number):
    
    rng = default_rng(seed=seed_number)

    # cols_shuffled_df = df.sample(frac=1, axis=1, random_state=np.random.RandomState(shuffle_cols))

    # compute probit (param) of a student answering a question correctly, for each student
    student_param_arr = train_df.sum(axis=1)/no_train_cols

    predictions_df = pd.DataFrame().reindex_like(test_df)

    for i, probability in enumerate(student_param_arr):
        predictions_df.iloc[i] = rng.binomial(size=len(test_df.columns), n=1, p=probability)

    total_entries = test_df.size

    no_correct_predictions = np.count_nonzero(predictions_df == test_df)
    performance = no_correct_predictions/total_entries

    print(performance*100)
    return performance

def train_question_baseline(train_df, test_df, seed_number):
    
    rng = default_rng(seed=seed_number)

    # compute probit of a question being answered correctly, for each question
    question_param_arr = train_df.sum(axis=0)/no_train_rows

    predictions_df = pd.DataFrame().reindex_like(test_df)
    for i, probability in enumerate(question_param_arr):
        predictions_df.iloc[:, i] = rng.binomial(size=len(test_df), n=1, p=probability)

    total_entries = test_df.size

    no_correct_predictions = np.count_nonzero(predictions_df == test_df)
    performance = no_correct_predictions/total_entries

    print(performance*100)
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

# performance_arr = [train_student_baseline(train_student_df, test_df, i)*100 for i in range(100)]
performance_arr = [train_question_baseline(train_question_df, test_df, i)*100 for i in range(100)]
print(performance_arr)
print(np.mean(performance_arr), np.std(performance_arr))

# elif metric == 'cf':
#     true_pos, false_pos, true_neg, false_neg = 0,0,0,0
#     for col in predictions_df:
#         temp_cf = pd.crosstab(predictions_df[col], test_df[col], rownames=['Actual'], colnames=['Predicted'])
#         true_pos += temp_cf.iloc[1,1]
#         false_pos += temp_cf.iloc[0,1]
#         true_neg += temp_cf.iloc[0,0]
#         false_neg += temp_cf.iloc[1,0]
#     performance = [true_pos/total_entries, false_pos/total_entries, true_neg/total_entries, false_neg/total_entries]
