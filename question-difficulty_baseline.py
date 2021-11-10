import torch
import numpy as np
import pandas as pd

from utils.clean_data import thres_score_range
from utils.binarise_data import binarise_by_mid, binarise_by_avg
from utils.shuffle_data import shuffle_cols
from utils.split_data import split_to_4quadrants

df = pd.read_csv("personalised-education/Fwd__Pinpoint_ML_Dataset/9to1_2017_GCSE_1H.csv", skiprows=list(range(1,24)), usecols=list(range(2,26)))
max_scores = pd.read_csv("personalised-education/Fwd__Pinpoint_ML_Dataset/9to1_2017_GCSE_1H.csv", nrows = 1, skiprows=[1], usecols=list(range(2,26)))

cleaned_df = thres_score_range(df, max_scores)
binarised_df = binarise_by_mid(cleaned_df, max_scores)
# binarised_df = binarise_by_avg(cleaned_df)
binarised_df = shuffle_cols(binarised_df)

dataset_ts = torch.tensor(binarised_df.values)
_, train_question_ts, _, test_ts = split_to_4quadrants(dataset_ts)

def train_question_baseline(train_ts, test_ts, seed_number):
    
    rng = torch.Generator()
    rng.manual_seed(seed_number)

    no_train_rows = train_ts.shape[0]
    question_param_arr = torch.sum(train_ts, dim=0, keepdim=True) / no_train_rows
    x = question_param_arr.repeat(no_train_rows,1)
    predictions = torch.bernoulli(x, generator=rng)

    performance = torch.sum(torch.eq(test_ts, predictions)) / torch.numel(test_ts)
    performance = float(performance)*100
    return performance

performance_arr = [train_question_baseline(train_question_ts, test_ts, i) for i in range(100)]
print(performance_arr)
print(np.mean(performance_arr), np.std(performance_arr))
