import pandas as pd
import torch

from utils.parse_data import parse_paper_data

from utils.clean_data import thres_score_range
from utils.binarise_data import binarise_by_mid, binarise_by_avg
from utils.shuffle_data import shuffle_cols
from utils.split_data import split_to_4quadrants, split_to_4quadrants_df

from baseline_models.single_param_baseline import train_single_param
from baseline_models.student_ability_baseline import train_student_ability
from baseline_models.question_difficulty_baseline import train_question_difficulty
from models.ADP_quadrant import train_product_alternate_quadrants, train_product_upper_left
from models.ADP_vectorised import train_product_vectorised
from models.meta_vectorised import train_product_meta
from models.meta_quadrant import train_product_upper_left_meta
from models.interactive import train_product_interactive

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
data_start_row = 6

raw_data = pd.read_csv(csv_path, low_memory=False)

exam_data_df, meta_data_df = parse_paper_data(raw_data, data_start_row, paper1_columns)
max_scores = meta_data_df.loc['Max'].astype(float)
exam_data_df = thres_score_range(exam_data_df, max_scores)
exam_data_df = binarise_by_mid(exam_data_df, max_scores)

exam_data_df2, meta_data_df2 = parse_paper_data(raw_data, data_start_row, paper2_columns)
max_scores2 = meta_data_df2.loc['Max'].astype(float)
exam_data_df2 = thres_score_range(exam_data_df2, max_scores2)
exam_data_df2 = binarise_by_mid(exam_data_df2, max_scores2)

combined_df = pd.concat([exam_data_df, exam_data_df2], axis=1)
combined_ts = torch.tensor(combined_df.values)

S, Q = combined_ts.shape[0], combined_ts.shape[1]
reshaped_data = combined_ts.reshape(-1) # unstack data

student_id = torch.arange(S)
student_id = student_id.repeat(Q, 1).T.reshape(-1)

question_id = torch.arange(Q)
question_id = question_id.repeat(S)

vectorised_data_ts = torch.stack((reshaped_data, student_id, question_id), dim=0)
vectorised_data_ts_cleaned = vectorised_data_ts.T[~torch.any(vectorised_data_ts.isnan(),dim=0)].type(torch.int).T

seed_number = 1000
rng = torch.Generator()
rng.manual_seed(seed_number)

train_product_vectorised(vectorised_data_ts_cleaned, combined_df, rng, learning_rate=0.00025, n_iters=31)