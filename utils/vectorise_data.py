import torch

def vectorise_data(data_ts, data_df):
    S, Q = data_ts.shape[0], data_ts.shape[1]
    
    reshaped_data = data_ts.reshape(-1).type(torch.int) # unstack data
    
    student_id = torch.tensor(data_df.index.values)
    student_id = student_id.repeat(Q, 1).T.reshape(-1)
    
    question_id = torch.tensor([int(entry[1:])-1 for entry in data_df.columns.tolist()])
    question_id = question_id.repeat(S)

    vectorised_data_ts = torch.stack((reshaped_data, student_id, question_id), dim=0)

    return vectorised_data_ts