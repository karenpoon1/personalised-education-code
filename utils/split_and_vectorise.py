import torch
import random

def split_into_train_test(data_ts, row_range, col_range):
    
    train_ts = torch.clone(data_ts)
    train_ts[row_range[0]:row_range[1], col_range[0]:col_range[1]] = float('nan') # Replace testset data value with 'nan'

    test_ts = torch.clone(data_ts[row_range[0]:row_range[1], col_range[0]:col_range[1]])

    return train_ts, test_ts

def vectorise_data(data_ts, student_id_range, question_id_range, shuffle=False):

    S, Q = data_ts.shape[0], data_ts.shape[1]
    reshaped_data = data_ts.reshape(-1).type(torch.float) # unstack data

    student_id = torch.arange(student_id_range[0], student_id_range[1])
    student_id = student_id.repeat(Q, 1).T.reshape(-1)

    question_id = torch.arange(question_id_range[0], question_id_range[1])
    question_id = question_id.repeat(S)

    vectorised_ts = torch.stack((reshaped_data, student_id, question_id), dim=0)
    vectorised_ts = vectorised_ts.T[~torch.any(vectorised_ts.isnan(),dim=0)].type(torch.int).T

    if shuffle:
        col_idxs = list(range(vectorised_ts.shape[1]))
        random.seed(1000)
        random.shuffle(col_idxs)
        vectorised_ts = vectorised_ts[:, torch.tensor(col_idxs)]

    return vectorised_ts

def split_and_vectorise(data_ts, row_range, col_range, shuffle=False):
    S, Q = data_ts.shape[0], data_ts.shape[1]

    train_ts, test_ts = split_into_train_test(data_ts, row_range, col_range)
    train_vectorised_ts = vectorise_data(train_ts, [0, S], [0, Q], shuffle)
    test_vectorised_ts = vectorise_data(test_ts, row_range, col_range, shuffle=False)
    
    return train_vectorised_ts, test_vectorised_ts
