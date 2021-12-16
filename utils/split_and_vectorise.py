import torch
import random

torch.manual_seed(1000)

def split_into_train_test(data_ts, test_range):
    S_range, Q_range = test_range[0], test_range[1]
    
    train_ts = torch.clone(data_ts)
    train_ts[S_range[0]:S_range[1], Q_range[0]:Q_range[1]] = float('nan')

    test_ts = torch.clone(data_ts[S_range[0]:S_range[1], Q_range[0]:Q_range[1]])

    return train_ts, test_ts

def vectorise_data(data_ts, data_range, shuffle=False):
    S_range, Q_range = data_range[0], data_range[1]

    S, Q = data_ts.shape[0], data_ts.shape[1]
    reshaped_data = data_ts.reshape(-1).type(torch.float) # unstack data

    student_id = torch.arange(S_range[0], S_range[1])
    student_id = student_id.repeat(Q, 1).T.reshape(-1)

    question_id = torch.arange(Q_range[0], Q_range[1])
    question_id = question_id.repeat(S)

    vectorised_ts = torch.stack((reshaped_data, student_id, question_id), dim=0)
    vectorised_ts = vectorised_ts.T[~torch.any(vectorised_ts.isnan(),dim=0)].type(torch.int).T

    if shuffle:
        col_idxs = list(range(vectorised_ts.shape[1]))
        random.seed(1000)
        random.shuffle(col_idxs)
        vectorised_ts = vectorised_ts[:, torch.tensor(col_idxs)]

    return vectorised_ts

def split_and_vectorise(data_ts, test_range, shuffle=False):
    S, Q = data_ts.shape[0], data_ts.shape[1]
    test_range = [[int(S*0.5), int(S)], [int(Q*0.5), int(Q)]]
    
    train_ts, test_ts = split_into_train_test(data_ts, test_range)
    train_vectorised_ts = vectorise_data(train_ts, [[0,S], [0,Q]], shuffle)
    test_vectorised_ts = vectorise_data(test_ts, test_range, shuffle=False)
    return train_vectorised_ts, test_vectorised_ts
