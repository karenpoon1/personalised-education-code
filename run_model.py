import torch
import numpy as np

from utils.clean_data import thres_score_range
from utils.binarise_data import binarise_by_mid, binarise_by_avg
from utils.shuffle_data import shuffle_cols
from utils.split_data import split_to_4quadrants

from models.single_param_baseline import train_single_param
from models.student_ability_baseline import train_student_ability
from models.question_difficulty_baseline import train_question_difficulty
from models.ability_difficulty_product import train_product_alternate_quadrants 
# from product_baseline_all_quadrants import train_product_all_quadrants
# from product_baseline_all_quadrants_draft import train_product_entire
# from product_diff_meta import train_meta

def run_model(exam_data_df, meta_data_df, model, binarise_method='mid', shuffle=False):

    max_scores = meta_data_df.loc['Max'].astype(float)
    cleaned_df = thres_score_range(exam_data_df, max_scores)

    if binarise_method == 'mid':
        binarised_df = binarise_by_mid(cleaned_df, max_scores)
    elif binarise_method == 'avg':
        binarised_df = binarise_by_avg(cleaned_df)
    
    if shuffle:
        binarised_df = shuffle_cols(binarised_df)

    dataset_ts = torch.tensor(binarised_df.values)
    first_quadrant_ts, train_question_ts, train_student_ts, test_ts = split_to_4quadrants(dataset_ts)

    if model == 'single_param':
        performance_arr = [train_single_param(first_quadrant_ts, train_question_ts, train_student_ts, test_ts, i) for i in range(100)]
        print(np.mean(performance_arr), np.std(performance_arr))

    elif model == 'student_ability':
        performance_arr = [train_student_ability(train_student_ts, test_ts, i) for i in range(100)]
        print(np.mean(performance_arr), np.std(performance_arr))

    elif model == 'question_difficulty':
        performance_arr = [train_question_difficulty(train_question_ts, test_ts, i) for i in range(100)]
        print(np.mean(performance_arr), np.std(performance_arr))

    elif model == 'ability_difficulty_product':
        seed_number = 1000
        rng = torch.Generator()
        rng.manual_seed(seed_number)
        train_product_alternate_quadrants(train_question_ts, train_student_ts, test_ts, 0.0003, 60, rng)
    
    return