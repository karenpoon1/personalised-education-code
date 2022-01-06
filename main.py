import pandas as pd

from utils.parse_data import parse_paper_data
from utils.preprocess_data import process_raw
from run_model import Models
from models.ADP import ADP
from models.SP import SP

paper1_columns = ['Name', 'q1', 'q2', 'q3',
                      'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11', 'q12', 'q13', 'q14',
                      'q15', 'q16', 'q17', 'q18', 'q19', 'q20', 'q21', 'q22', 'q23', 'q24']

# old paper
paper_csv = "Fwd__Pinpoint_ML_Dataset/9to1_2017_GCSE_1H.csv"
paper_start = 23

raw_data = pd.read_csv(paper_csv, low_memory=False)
exam_data_df, meta_data_df = parse_paper_data(raw_data, paper_start, paper1_columns)

# # new paper
# paper_csv = "Fwd__Pinpoint_ML_Dataset\9to1_2017_GCSE_1H_and_2H_and_3H Linked Pinpoint Data_Cleaned.csv"
# paper_start = 6

# raw_data = pd.read_csv(paper_csv, low_memory=False)
# exam_data_df, meta_data_df = parse_paper_data(raw_data, paper_start, paper1_columns)

old_data_models = Models(exam_data_df, meta_data_df)
# old_data_models.ADP_quadrant(iters=15000)
old_data_models.ADP_vectorised(learning_rate=0.0002, iters=5930)
# old_data_models.meta(iters=1530)
# old_data_models.interactive(learning_rate=0.0002 ,iters=6000)

# # new_data_models = Models(new_exam_data_df, new_meta_data_df)
# # new_data_models.ADP_vectorised(iters=20000)
# # new_data_models.ADP_vectorised(iters=4550)
# # new_data_models.meta(iters=1520)
# # new_data_models.interactive(iters=2)

# _, new_exam_data_ts = process_raw(new_exam_data_df, new_meta_data_df, binarise_method='mid', shuffle=True)

# S, Q = new_exam_data_ts.shape[0], new_exam_data_ts.shape[1] # Data block size
# my_ADP = ADP(new_exam_data_ts, [int(S/2), S], [int(Q/2), Q])
# my_ADP.run(learning_rate=0.00025, iters=100)
