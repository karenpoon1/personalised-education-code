import torch
import math

def split_to_4quadrants(dataset_ts, student_split=0.5, question_split=0.5):

    # split to train and test set
    no_train_rows = math.ceil(dataset_ts.shape[0] * student_split)
    no_train_cols = math.ceil(dataset_ts.shape[1] * question_split)

    upper_half_ts, lower_half_ts = torch.split(dataset_ts, no_train_rows, dim=0)
    first_quadrant_ts, train_question_ts = torch.split(upper_half_ts, no_train_cols, dim=1)
    train_student_ts, test_ts = torch.split(lower_half_ts, no_train_cols, dim=1)

    return first_quadrant_ts, train_question_ts, train_student_ts, test_ts


def split_to_4quadrants_df(dataset_df, student_split=0.5, question_split=0.5):
    
    no_train_rows = int(len(dataset_df) * student_split)
    no_train_cols = int(len(dataset_df.columns) * question_split)

    first_quadrant_df = dataset_df.iloc[:no_train_rows, :no_train_cols]
    train_question_df = dataset_df.iloc[:no_train_rows, no_train_cols:]
    train_student_df = dataset_df.iloc[no_train_rows:, :no_train_cols]
    test_df = dataset_df.iloc[no_train_rows:, no_train_cols:]

    return first_quadrant_df, train_question_df, train_student_df, test_df
