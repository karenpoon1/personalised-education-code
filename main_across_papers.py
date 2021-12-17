import pandas as pd
import numpy as np
import torch

from models.SP import SP
from models.ADP import ADP

from utils.parse_data import parse_paper_data
from utils.preprocess_data import process_raw

paper1_columns = ['Name', 'q1', 'q2', 'q3',
                      'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11', 'q12', 'q13', 'q14',
                      'q15', 'q16', 'q17', 'q18', 'q19', 'q20', 'q21', 'q22', 'q23', 'q24']
paper2_columns = ['Name.1', 'q1.1', 'q2.1', 'q3.1',
                    'q4.1', 'q5.1', 'q6.1', 'q7.1', 'q8.1', 'q9.1', 'q10.1', 'q11.1',
                    'q12.1', 'q13.1', 'q14.1', 'q15.1', 'q16.1', 'q17.1', 'q18.1', 'q19.1',
                    'q20.1', 'q21.1', 'q22.1', 'q23.1']
paper3_columns = ['Name.2', 'q1.2', 'q2.2', 'q3.2', 'q4.2', 'q5.2', 'q6.2',
                    'q7.2', 'q8.2', 'q9.2', 'q10.2', 'q11.2', 'q12.2', 'q13.2', 'q14.2',
                    'q15.2', 'q16.2', 'q17.2', 'q18.2', 'q19.2', 'q20.2', 'q21.2', 'q22.2',
                    'q23.2']

csv_path = "Fwd__Pinpoint_ML_Dataset\9to1_2017_GCSE_1H_and_2H_and_3H Linked Pinpoint Data_Cleaned.csv"
raw_data = pd.read_csv(csv_path, low_memory=False)
data_start_row = 6

# Process data
exam_data_df, meta_data_df = parse_paper_data(raw_data, data_start_row, paper1_columns)
exam_data_df, _ = process_raw(exam_data_df, meta_data_df, binarise_method='mid', shuffle=True)

exam_data_df2, meta_data_df2 = parse_paper_data(raw_data, data_start_row, paper2_columns)
exam_data_df2, _ = process_raw(exam_data_df2, meta_data_df2, binarise_method='mid', shuffle=True)

combined_df = pd.concat([exam_data_df, exam_data_df2], axis=1)
combined_ts = torch.clone(torch.tensor(combined_df.values))

S, Q = combined_ts.shape[0], combined_ts.shape[1]

# Run model
my_SP = SP(combined_ts, [0, 15000], [30, Q])
my_SP.mass_run()

# my_ADP = ADP(combined_ts, [0, 15000], [30, Q])
# my_ADP.run(learning_rate=0.00025, iters=10000)
