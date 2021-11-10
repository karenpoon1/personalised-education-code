import torch

def split_to_4quadrants(dataset_ts, student_split=0.5, question_split=0.5):

    # split to train and test set
    no_train_rows = int(dataset_ts.shape[0] * student_split)
    no_train_cols = int(dataset_ts.shape[1] * question_split)

    upper_half_ts, lower_half_ts = torch.split(dataset_ts, no_train_rows, dim=0)
    first_quadrant_ts, train_question_ts = torch.split(upper_half_ts, no_train_cols, dim=1)
    train_student_ts, test_ts = torch.split(lower_half_ts, no_train_cols, dim=1)

    return first_quadrant_ts, train_question_ts, train_student_ts, test_ts
