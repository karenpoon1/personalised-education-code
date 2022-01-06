import torch
import random

from torch.functional import split

from utils.clean_data import thres_score_range
from utils.binarise_data import binarise_by_mid, binarise_by_avg
from utils.shuffle_data import shuffle_cols
from utils.split_data import split_to_4quadrants, split_to_4quadrants_df
from utils.vectorise_data import vectorise_data, vectorise_unstructured_data
from utils.split_and_vectorise import split_and_vectorise

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
        train_vectorised_ts, test_vectorised_ts = split_and_vectorise(self.data_ts)

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
        test_ts = torch.clone(self.data_ts[S_start:S_end:, Q_start:Q_end])
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


    def interactive(self, learning_rate=0.00025, iters=5000, binarise_method='mid', shuffle='True'):
        self.process_raw(binarise_method, shuffle)
        self.set_rng(1000)
        acc, conf_matrix = train_interactive(self.data_ts, self.data_df, self.rng, learning_rate, iters)
        print(f"interactive (rate={learning_rate}, iters={iters}, binarise={binarise_method}, shuffle={shuffle}) -> accuracy: {acc}, confusion matrix: \n{conf_matrix}")
