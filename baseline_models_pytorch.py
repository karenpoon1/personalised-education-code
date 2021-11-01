import torch
import numpy as np
from numpy.random import default_rng
import pandas as pd



df = pd.read_csv("personalised-education/Fwd__Pinpoint_ML_Dataset/9to1_2017_GCSE_1H.csv", skiprows=list(range(1,24)), usecols=list(range(2,26)))
max_scores = pd.read_csv("personalised-education/Fwd__Pinpoint_ML_Dataset/9to1_2017_GCSE_1H.csv", nrows = 1, skiprows=[1], usecols=list(range(2,26)))

# clean data (scores above max go to max)
for col in df:
    max_score = max_scores[col].iloc[0]
    df.loc[df[col] > max_score, col] = max_score
    df.loc[df[col] < 0, col] = 0

# binarise data using average
for col in df:
    mean_score = df[col].mean()
    df[col] = (df[col] >= mean_score).astype(float)

# split to train and test set
student_split = 0.5
question_split = 0.5
no_train_rows = int(len(df) * student_split)
no_train_cols = int(len(df.columns) * question_split)

# shuffle data
# shuffle_seed = 0
# df = df.sample(frac=1, axis=0, random_state=np.random.RandomState(shuffle_seed)) # shuffle rows
# df = df.sample(frac=1, axis=1, random_state=np.random.RandomState(shuffle_seed)) # shuffle cols
# df = df.reset_index(drop=True)

train_question_df = df.iloc[:no_train_rows, no_train_cols:]
train_question_df = train_question_df.reset_index(drop=True)
train_student_df = df.iloc[no_train_rows:, :no_train_cols]
test_df = df.iloc[no_train_rows:, no_train_cols:]


def train_student_baseline(train_student_df, test_df, seed_number):

    rng = torch.Generator()
    rng.manual_seed(seed_number)

    train_student_tensor = torch.tensor(train_student_df.values)
    test_tensor = torch.tensor(test_df.values)

    student_param_tensor = torch.sum(train_student_tensor, dim=1, keepdim=True) / no_train_cols

    x = student_param_tensor.repeat(1,12)
    predictions = torch.bernoulli(x, generator=rng)

    performance = torch.sum(torch.eq(test_tensor, predictions)) / torch.numel(test_tensor)
    performance = float(performance)*100

    print(performance)
    return performance

def train_question_baseline(train_df, test_df, seed_number):
    
    rng = default_rng(seed=seed_number)

    question_param_arr = train_df.sum(axis=0)/no_train_rows

    predictions_df = pd.DataFrame().reindex_like(test_df)
    for i, probability in enumerate(question_param_arr):
        predictions_df.iloc[:, i] = rng.binomial(size=len(test_df), n=1, p=probability)

    no_correct_predictions = np.count_nonzero(predictions_df == test_df)
    performance = no_correct_predictions/test_df.size
    print(performance*100)

    return performance

performance_arr = [train_student_baseline(train_student_df, test_df, i) for i in range(100)]
print(performance_arr)
print(np.mean(performance_arr), np.std(performance_arr))

# performance_arr = [train_question_baseline(train_question_df, test_df, i)*100 for i in range(100)]
# print(performance_arr)
# print(np.mean(performance_arr), np.std(performance_arr))
