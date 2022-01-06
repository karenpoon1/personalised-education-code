import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from utils.preprocess_data import process_raw
from utils.split_and_vectorise import split_and_vectorise

class SP:
    def __init__(self, data_ts, testset_row_range, testset_col_range) -> None:
        self.data_ts = data_ts
        
        self.testset_row_range = testset_row_range
        self.testset_col_range = testset_col_range


    def run(self, seed_number=1000, validation=0.9):
        rng = torch.Generator()
        rng.manual_seed(seed_number)

        train_vectorised_ts, test_vectorised_ts = split_and_vectorise(self.data_ts, self.testset_row_range, self.testset_col_range)
        # train_vectorised_ts, _ = torch.split(train_vectorised_ts, int(torch.numel(train_vectorised_ts[0])*validation), dim=1)

        ones_train = torch.sum(train_vectorised_ts[0])
        total_train = torch.numel(train_vectorised_ts[0])
        single_param = ones_train / total_train

        predicted_probit = single_param.repeat(1, test_vectorised_ts.shape[1])
        predictions = torch.bernoulli(predicted_probit, generator=rng)

        performance = torch.sum(torch.eq(test_vectorised_ts[0], predictions)) / torch.numel(test_vectorised_ts[0])
        performance = float(performance)*100

        return performance

    def mass_run(self):
        acc_arr = [self.run(i) for i in range(100)]
        acc_mean, acc_std = np.mean(acc_arr), np.std(acc_arr)
        print(f"SP -> acc mean: {acc_mean}, acc std: {acc_std}")
