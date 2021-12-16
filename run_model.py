import torch
import random

from utils.clean_data import thres_score_range
from utils.binarise_data import binarise_by_mid, binarise_by_avg
from utils.shuffle_data import shuffle_cols
from utils.split_data import split_to_4quadrants
from utils.vectorise_data import vectorise_unstructured_data

from models.single_param_baseline import train_single_param
from models.student_ability_baseline import train_student_ability
from models.question_difficulty_baseline import train_question_difficulty
from models.ADP_quadrant import train_product_alternate_quadrants, train_product_upper_left
from models.ADP_vectorised import train_product_vectorised
# from models.meta_quadrant import train, train_product_upper_left_meta
from models.meta_vectorised import train_meta_vectorised
from models.interactive import train_interactive

class Models:
    def __init__(self, raw_data_df, raw_meta_df) -> None:
        self.raw_data_df = raw_data_df
        self.raw_meta_df = raw_meta_df
        self.max_scores = raw_meta_df.loc['Max'].astype(float)


    def process_raw(self, binarise_method, shuffle):
        thres_df = thres_score_range(self.raw_data_df, self.max_scores)
        if binarise_method == 'mid':
            binarised_df = binarise_by_mid(thres_df, self.max_scores)
        elif binarise_method == 'avg':
            binarised_df = binarise_by_avg(thres_df)
        
        if shuffle:
            binarised_df = shuffle_cols(binarised_df)

        self.data_df = binarised_df
        self.data_ts = torch.tensor(binarised_df.values)


    def set_rng(self, seed_number):
        rng = torch.Generator()
        rng.manual_seed(seed_number)
        self.rng = rng


    def single_param(self, binarise_method='mid', shuffle='True'):
        self.process_raw(binarise_method, shuffle)
        first_quadrant_ts, train_question_ts, train_student_ts, test_ts = split_to_4quadrants(self.data_ts)
        acc_mean, acc_std = train_single_param(first_quadrant_ts, train_question_ts, train_student_ts, test_ts)
        print(f"Single Parameter (binarise={binarise_method}, shuffle={shuffle}) -> mean: {acc_mean}, std: {acc_std}")


    def student_ability(self, binarise_method='mid', shuffle='True'):
        self.process_raw(binarise_method, shuffle)
        _, _, train_student_ts, test_ts = split_to_4quadrants(self.data_ts)
        acc_mean, acc_std = train_student_ability(train_student_ts, test_ts)
        print(f"Student Ability (binarise={binarise_method}, shuffle={shuffle}) -> mean: {acc_mean}, std: {acc_std}")


    def question_difficulty(self, binarise_method='mid', shuffle='True'):
        self.process_raw(binarise_method, shuffle)
        _, train_question_ts, _, test_ts = split_to_4quadrants(self.data_ts)
        acc_mean, acc_std = train_question_difficulty(train_question_ts, test_ts)
        print(f"Question Difficulty (binarise={binarise_method}, shuffle={shuffle}) -> mean: {acc_mean}, std: {acc_std}")


    def ADP_quadrant(self, learning_rate=0.0002, iters=10000, binarise_method='mid', shuffle='True'):
        self.process_raw(binarise_method, shuffle)
        self.set_rng(1000)
        first_quadrant_ts, train_question_ts, train_student_ts, test_ts = split_to_4quadrants(self.data_ts)
        # acc, conf_matrix = train_product_alternate_quadrants(self.train_question_ts, self.train_student_ts, self.test_ts, 0.0003, 600, rng)
        acc, conf_matrix = train_product_upper_left(first_quadrant_ts, train_question_ts, train_student_ts, test_ts, learning_rate, iters, self.rng)
        print(f"ADP by quadrant (rate={learning_rate}, iters={iters}, binarise={binarise_method}, shuffle={shuffle}) -> accuracy: {acc}, confusion matrix: {conf_matrix}")


    def ADP_vectorised(self, learning_rate=0.00025, iters=3600, binarise_method='mid', shuffle='True'):
        self.process_raw(binarise_method, shuffle)
        self.set_rng(1000)
        S, Q = self.data_df.shape[0], self.data_df.shape[1] # Data block size

        # Split train and test set
        S_start, S_end, Q_start, Q_end = int(S*0.5), int(S), int(Q*0.5), int(Q)
        train_ts = torch.clone(self.data_ts)
        train_ts[S_start:S_end, Q_start:Q_end] = float('nan')
        test_ts = torch.clone(self.data_ts[S_start:, Q_start:])
        Q_test = test_ts.shape[1]
        # Vectorise train and test set
        train_vectorised_ts = vectorise_unstructured_data(train_ts, [0,S], [0,Q], shuffle=True)
        test_vectorised_ts = vectorise_unstructured_data(test_ts, [S_start, S_end], [Q_start, Q_end], shuffle=False)

        acc, conf_matrix = train_product_vectorised(train_vectorised_ts, test_vectorised_ts, S, Q, self.rng, learning_rate, iters, Q_test)
        print(f"ADP vectorised (rate={learning_rate}, iters={iters}, binarise={binarise_method}, shuffle={shuffle}) -> accuracy: {acc}, confusion matrix: \n{conf_matrix}")


    def meta(self, learning_rate=0.00025, iters=1800, binarise_method='mid', shuffle='True'):
        self.process_raw(binarise_method, shuffle)
        self.set_rng(1000)
        S, Q = self.data_df.shape[0], self.data_df.shape[1] # Data block size

        # Split train and test set
        S_start, S_end, Q_start, Q_end = int(S*0.5), int(S), int(Q*0.5), int(Q)
        train_ts = torch.clone(self.data_ts)
        train_ts[S_start:S_end, Q_start:Q_end] = float('nan')
        test_ts = torch.clone(self.data_ts[S_start:, Q_start:])
        Q_test = test_ts.shape[1]
        # Vectorise train and test set
        train_vectorised_ts = vectorise_unstructured_data(train_ts, [0,S], [0,Q], shuffle=True)
        test_vectorised_ts = vectorise_unstructured_data(test_ts, [S_start, S_end], [Q_start, Q_end], shuffle=False)

        prob_solv_df = self.raw_meta_df.loc['Difficulty'].astype(float)
        prob_solv_ts = torch.tensor(prob_solv_df.values)

        acc, conf_matrix = train_meta_vectorised(train_vectorised_ts, test_vectorised_ts, S, Q, prob_solv_ts, self.rng, learning_rate, iters, Q_test)
        print(f"meta (rate={learning_rate}, iters={iters}, binarise={binarise_method}, shuffle={shuffle}) -> accuracy: {acc}, confusion matrix: \n{conf_matrix}")


    # def meta_quadrant(self, learning_rate=0.0002, iters=4500, binarise_method='mid', shuffle='False'):
    #     self.process_raw(binarise_method, shuffle)
    #     self.set_rng(1000)
    #     prob_solv_df = self.raw_meta_df.loc['Difficulty'].astype(float)
    #     prob_solv_ts = torch.tensor(prob_solv_df.values)
    #     train_product_upper_left_meta(self.first_quadrant_ts, self.train_question_ts, self.train_student_ts, self.test_ts, prob_solv_ts, self.rng, learning_rate, iters)


    def interactive(self, learning_rate=0.00025, iters=5000, binarise_method='mid', shuffle='False'):
        self.process_raw(binarise_method, shuffle)
        self.set_rng(1000)
        prob_solv_df = self.raw_meta_df.loc['Difficulty'].astype(float)
        prob_solv_ts = torch.tensor(prob_solv_df.values)
        acc, conf_matrix = train_interactive(self.data_ts, self.data_df, prob_solv_ts, self.rng, learning_rate, iters, held_out_test=True)
        print(f"interactive (rate={learning_rate}, iters={iters}, binarise={binarise_method}, shuffle={shuffle}) -> accuracy: {acc}, confusion matrix: \n{conf_matrix}")
