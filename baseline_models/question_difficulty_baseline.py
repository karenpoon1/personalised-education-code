import torch
import numpy as np

def train(train_ts, test_ts, seed_number):
    rng = torch.Generator()
    rng.manual_seed(seed_number)

    no_train_rows = train_ts.shape[0]
    question_param_arr = torch.sum(train_ts, dim=0, keepdim=True) / no_train_rows
    
    no_test_rows = test_ts.shape[0]
    x = question_param_arr.repeat(no_test_rows,1)
    predictions = torch.bernoulli(x, generator=rng)

    performance = torch.sum(torch.eq(test_ts, predictions)) / torch.numel(test_ts)
    performance = float(performance)*100
    return performance

def train_question_difficulty(train_question_ts, test_ts):
    performance_arr = [train(train_question_ts, test_ts, i) for i in range(100)]
    return np.mean(performance_arr), np.std(performance_arr)
