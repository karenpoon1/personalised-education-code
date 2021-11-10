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
# binarised_df = binarise_by_mid(cleaned_df, max_scores)
binarised_df = binarise_by_avg(cleaned_df)
binarised_df = shuffle_cols(binarised_df)

dataset_ts = torch.tensor(binarised_df.values)
first_quadrant_ts, train_question_ts, train_student_ts, test_ts = split_to_4quadrants(dataset_ts)

def train_single_param_baseline(first_quadrant_ts, train_question_ts, train_student_ts, test_ts, seed_number):

    rng = torch.Generator()
    rng.manual_seed(seed_number)

    no_ones_train = torch.sum(first_quadrant_ts) + torch.sum(train_question_ts) + torch.sum(train_student_ts)
    no_entries_train = torch.numel(first_quadrant_ts) + torch.numel(train_question_ts) + torch.numel(train_student_ts)
    single_param = no_ones_train / no_entries_train

    x = single_param.repeat(test_ts.shape[0], test_ts.shape[1])
    predictions = torch.bernoulli(x, generator=rng)

    performance = torch.sum(torch.eq(test_ts, predictions)) / torch.numel(test_ts)
    performance = float(performance)*100
    return performance

performance_arr = [train_single_param_baseline(first_quadrant_ts, train_question_ts, train_student_ts, test_ts, i) for i in range(100)]
print(performance_arr)
print(np.mean(performance_arr), np.std(performance_arr))
