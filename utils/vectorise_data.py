import torch
import random

def vectorise_data(data_ts, data_df): # Requires index reset in parse_data
    S, Q = data_ts.shape[0], data_ts.shape[1]
    reshaped_data = data_ts.reshape(-1).type(torch.int) # unstack data
    
    student_id = torch.tensor(data_df.index.values)
    student_id = student_id.repeat(Q, 1).T.reshape(-1)
    
    question_id = torch.tensor([int(entry[1:])-1 for entry in data_df.columns.tolist()])
    question_id = question_id.repeat(S)

    vectorised_data_ts = torch.stack((reshaped_data, student_id, question_id), dim=0)
    return vectorised_data_ts

def vectorise_unstructured_data(data_ts, S_range, Q_range, shuffle):
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
