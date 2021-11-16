import pandas as pd

from parse_data import parse_paper_data
from run_model import run_model

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

# old paper
csv_path = "personalised-education/Fwd__Pinpoint_ML_Dataset/9to1_2017_GCSE_1H.csv"
data_start_row = 23

# new paper
# csv_path = "personalised-education-public\Fwd__Pinpoint_ML_Dataset\9to1_2017_GCSE_1H_and_2H_and_3H Linked Pinpoint Data_Cleaned.csv"
# data_start_row = 6

model = ['single_param', 'student_ability', 'question_difficulty', 'product']

raw_data = pd.read_csv(csv_path, low_memory=False)
exam_data_df, meta_data_df = parse_paper_data(raw_data[paper1_columns], data_start_row)
run_model(exam_data_df, meta_data_df, model[1], binarise_method='mid', shuffle=True)
