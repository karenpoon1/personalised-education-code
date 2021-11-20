import torch
import pandas as pd
import matplotlib.pyplot as plt

from utils.clean_data import thres_score_range
from utils.binarise_data import binarise_by_avg, binarise_by_mid
from utils.shuffle_data import shuffle_cols
from utils.split_data import split_to_4quadrants

df = pd.read_csv("personalised-education/Fwd__Pinpoint_ML_Dataset/9to1_2017_GCSE_1H.csv", skiprows = [i for i in range(1,24)], usecols=[i for i in range(2,26)])
max_scores = pd.read_csv("personalised-education/Fwd__Pinpoint_ML_Dataset/9to1_2017_GCSE_1H.csv", nrows = 1, skiprows=[1], usecols=list(range(2,26)))

cleaned_df = thres_score_range(df, max_scores)
binarised_df = binarise_by_avg(cleaned_df)
# binarised_df = binarise_by_mid(cleaned_df, max_scores)
# binarised_df = shuffle_cols(binarised_df)

dataset_ts = torch.tensor(binarised_df.values)
first_quadrant, train_question_ts, train_student_ts, _ = split_to_4quadrants(dataset_ts)
first_quadrant_df, train_question_df, train_student_df = pd.DataFrame(first_quadrant), pd.DataFrame(train_question_ts), pd.DataFrame(train_student_ts)

question_probit = train_question_df.sum(axis=0)/train_question_df.shape[0]
question_probit.plot.bar()
plt.title('Bar plot of question parameter')
plt.show()

question_probit.plot.hist()
plt.title('Histogram of question parameter (last 12q)')
plt.show()

first_12_question_probit = first_quadrant_df.sum(axis=0)/first_quadrant_df.shape[0]
first_12_question_probit.plot.bar()
plt.title('Bar plot of question parameter (first 12q)')
plt.show()

first_12_question_probit.plot.hist()
plt.title('Histogram of question parameter (first 12q)')
plt.show()

# compute probit of a student answering a question correctly, for each student
student_probit = train_student_df.sum(axis=1)/train_student_df.shape[1]
student_probit.plot.hist(bins=train_student_df.shape[1]+1)
plt.title('Histogram of student parameter')
plt.show()
