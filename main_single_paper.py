import pandas as pd

from utils.parse_data import parse_paper_data
from run_model import Models

# old paper
old_paper_csv = "Fwd__Pinpoint_ML_Dataset/9to1_2017_GCSE_1H.csv"
old_paper_start = 23

# new paper
new_paper_csv = "Fwd__Pinpoint_ML_Dataset\9to1_2017_GCSE_1H_and_2H_and_3H Linked Pinpoint Data_Cleaned.csv"
new_paper_start = 6

paper1_columns = ['Name', 'q1', 'q2', 'q3',
                      'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11', 'q12', 'q13', 'q14',
                      'q15', 'q16', 'q17', 'q18', 'q19', 'q20', 'q21', 'q22', 'q23', 'q24']

raw_data = pd.read_csv(new_paper_csv, low_memory=False)
exam_data_df, meta_data_df = parse_paper_data(raw_data, new_paper_start, paper1_columns)

my_models = Models(exam_data_df, meta_data_df)
# my_models.ADP_vectorised(iters=4550)
# my_models.meta(iters=1520)
my_models.interactive(shuffle=True, iters=100)

# my_models.ADP_vectorised(learning_rate=0.0002, iters=5930)
# my_models.meta(iters=1530)