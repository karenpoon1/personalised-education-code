import pandas as pd

from utils.split_data import split_to_4quadrants
from utils.preprocess_data import process_raw
from utils.parse_data import parse_paper_data

from baseline_models.single_param_baseline import train_single_param
from baseline_models.student_ability_baseline import train_student_ability
from baseline_models.question_difficulty_baseline import train_question_difficulty

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

binarise_method, shuffle, student_split, question_split = 'mid', 'True', 0.5, 0.5

_, data_ts = process_raw(exam_data_df, meta_data_df, binarise_method, shuffle)
first_quadrant_ts, train_question_ts, train_student_ts, test_ts = split_to_4quadrants(data_ts, student_split, question_split)
print(f"Baseline models (binarise_method={binarise_method}, shuffle={shuffle}, student_split={student_split}, question_split={question_split})")

SP_acc_mean, SP_acc_std = train_single_param(first_quadrant_ts, train_question_ts, train_student_ts, test_ts)
print(f"Single Parameter -> mean: {SP_acc_mean}, std: {SP_acc_std}")

SA_acc_mean, SA_acc_std = train_student_ability(train_student_ts, test_ts)
print(f"Student Ability -> mean: {SA_acc_mean}, std: {SA_acc_std}")

QD_acc_mean, QD_acc_std = train_question_difficulty(train_question_ts, test_ts)
print(f"Question Difficulty -> mean: {QD_acc_mean}, std: {QD_acc_std}")
