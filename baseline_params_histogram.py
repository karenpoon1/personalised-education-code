import numpy as np
from numpy.random import default_rng
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("personalised-education/Fwd__Pinpoint_ML_Dataset/9to1_2017_GCSE_1H.csv", skiprows = [i for i in range(1,24)], usecols=[i for i in range(2,26)])
max_scores = pd.read_csv("personalised-education/Fwd__Pinpoint_ML_Dataset/9to1_2017_GCSE_1H.csv", nrows = 1, skiprows=[1], usecols=list(range(2,26)))

# clean data (scores above max go to max)
for col in df:
    max_score = max_scores[col].iloc[0]
    df.loc[df[col] > max_score, col] = max_score
    df.loc[df[col] < 0, col] = 0
    # binarise data by thresholding at mid score
    # df[col] = (df[col] >= max_score/2).astype(float)

    # binarise data by thresholding at average
    mean_score = df[col].mean()
    df[col] = (df[col] >= mean_score).astype(float)

def display_parameter_histogram(student_split, question_split, df, shuffle_rows, shuffle_cols):
    
    seed_number = 0
    if shuffle_rows:
        df = df.sample(frac=1, axis=0, random_state=np.random.RandomState(seed_number))
        df = df.reset_index(drop=True)
    if shuffle_cols:
        df = df.sample(frac=1, axis=1, random_state=np.random.RandomState(seed_number))

    # split to train and test set
    no_train_rows = int(len(df) * student_split)
    no_train_cols = int(len(df.columns) * question_split)

    train_question_df = df.iloc[:no_train_rows, no_train_cols:]
    train_question_df = train_question_df.reset_index(drop=True)
    train_student_df = df.iloc[no_train_rows:, :no_train_cols]
    test_df = df.iloc[no_train_rows:, no_train_cols:]
    redundant_df = df.iloc[:no_train_rows, :no_train_cols]

    # compute probit of a question being answered correctly, for each question
    question_probit = train_question_df.sum(axis=0)/no_train_rows
    question_probit.plot.bar()
    plt.title('Bar plot of question parameter (last 12q)')
    plt.show()

    question_probit.plot.hist()
    plt.title('Hist plot of question parameter (last 12q)')
    plt.show()

    redundant_question_probit = redundant_df.sum(axis=0)/no_train_rows
    redundant_question_probit.plot.hist()
    plt.title('Hist plot of question parameter (first 12q)')
    plt.show()

    redundant_question_probit.plot.bar()
    plt.title('Bar plot of question parameter (first 12q)')
    plt.show()

    # compute probit of a student answering a question correctly, for each student
    student_probit = train_student_df.sum(axis=1)/no_train_cols
    student_probit.plot.hist(bins=no_train_cols+1)
    plt.title('Histogram of student parameter')
    plt.show()

display_parameter_histogram(0.5, 0.5, df, shuffle_rows=False, shuffle_cols=False)