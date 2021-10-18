import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# extract properties of columns from .csv

df = pd.read_csv("personalised-education/Fwd__Pinpoint_ML_Dataset/9to1_2017_GCSE_1H.csv", skiprows = [i for i in range(1,24)], usecols=[i for i in range(2,26)])

for col in df:
    mean_score = df[col].mean()
    df[col] = (df[col] >= mean_score).astype(float) # binarise data

no_samples = len(df)
student_baseline_df = df.head(no_samples)

# shuffle rows
student_baseline_df = student_baseline_df.sample(frac=1)
student_baseline_df = student_baseline_df.reset_index(drop=True)

# split to train and test set
validation_split = 0.5
no_train_rows = int(no_samples * validation_split)
train_df = student_baseline_df.iloc[:no_train_rows]
test_df = student_baseline_df.iloc[no_train_rows:]

# compute probit of a question being answered correctly, for each question
question_probit = train_df.sum(axis=0)/no_train_rows

predictions_df = pd.DataFrame().reindex_like(test_df)

no_test_rows = no_samples - no_train_rows
for i in range(len(question_probit)):
    predictions_df.iloc[:, i] = np.random.binomial(size=no_test_rows, n=1, p=question_probit[i])

no_correct_predictions = np.count_nonzero(predictions_df == test_df)
total_entries = test_df.shape[0]*test_df.shape[1]
performance = no_correct_predictions/total_entries

print(performance)
