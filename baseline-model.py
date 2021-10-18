import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# extract properties of columns from .csv

df = pd.read_csv("../Fwd__Pinpoint_ML_Dataset/9to1_2017_GCSE_1H.csv", skiprows = [i for i in range(1,24)], usecols=[i for i in range(2,26)])

for col in df:
    mean_score = df[col].mean()
    df[col] = (df[col] >= mean_score).astype(float) # binarise data

# train, test = train_test_split(df, test_size=0.9, random_state=42, shuffle=True)

# shuffled_df = df.reindex(np.random.permutation(df.index))
student_baseline_df = df.head(5000)

# shuffle rows and columns
student_baseline_df = student_baseline_df.sample(frac=1, axis=1).sample(frac=1)
# reset row index (from student 0 to end, columns remain shuffled)
student_baseline_df = student_baseline_df.reset_index(drop=True)

# split to train and test set
no_train_cols = 12
no_test_cols = 12
train_df = student_baseline_df.iloc[:, :no_train_cols]
test_df = student_baseline_df.iloc[:, no_train_cols:]

# compute probit of a student answering a question correctly, for each student
student_probit = train_df.sum(axis=1)/no_train_cols
print(student_probit)

predictions_df = pd.DataFrame().reindex_like(test_df)

for i in range(len(student_probit)):
    predictions_df.loc[i] = np.random.binomial(size=no_test_cols, n=1, p=student_probit[i])

print(predictions_df)
print(test_df)