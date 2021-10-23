import numpy as np
from numpy.random import default_rng
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

df = pd.read_csv("personalised-education/Fwd__Pinpoint_ML_Dataset/9to1_2017_GCSE_1H.csv", skiprows = [i for i in range(1,24)], usecols=[i for i in range(2,26)])

# binarise data using average
for col in df:
    mean_score = df[col].mean()
    df[col] = (df[col] >= mean_score).astype(float)

# correlation matrix between questions
corr_matrix = df.corr()
sn.heatmap(corr_matrix, annot=True)
plt.show()

# correlation matrix between students
seed_number = 0
no_samples = 10

shuffled_df = df.sample(frac=1, random_state=np.random.RandomState(seed_number))
shuffled_df = shuffled_df.reset_index(drop=True)

rng = default_rng(seed=seed_number)
frac_df = shuffled_df.head(no_samples)
transposed_df = frac_df.transpose()

corr_matrix = transposed_df.corr()
sn.heatmap(corr_matrix, annot=True)
plt.show()
