import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from utils.split_and_vectorise import split_and_vectorise

class ADP:
    def __init__(self, data_ts, testset_row_range, testset_col_range, binarise_method='mid', shuffle=True, seed_number=1000) -> None:
        self.data_ts = data_ts
        
        self.testset_row_range = testset_row_range
        self.testset_col_range = testset_col_range

        self.binarise_method = binarise_method
        self.shuffle = shuffle

        rng = torch.Generator()
        rng.manual_seed(seed_number)
        self.rng = rng

    @staticmethod
    def probit_correct(bs, bq):
        return 1/(1+torch.exp(-bs-bq))

    def run(self, learning_rate, iters, shuffle=False):
        train_vectorised_ts, test_vectorised_ts = split_and_vectorise(self.data_ts, self.testset_row_range, self.testset_col_range, shuffle)
        
        S, Q = self.data_ts.shape[0], self.data_ts.shape[1] # Data block size
        trained_bs, trained_bq, nll_train_arr, nll_test_arr, acc_arr = self.train(train_vectorised_ts, test_vectorised_ts, S, Q, learning_rate, iters)
        accuracy, conf_matrix = self.predict(trained_bs, trained_bq, test_vectorised_ts, vis=True)

        self.plot_result(nll_train_arr/train_vectorised_ts.shape[1], nll_test_arr/test_vectorised_ts.shape[1], acc_arr, iters)
        print(f"ADP vectorised (rate={learning_rate}, iters={iters}, binarise={self.binarise_method}, shuffle={self.shuffle}) -> accuracy: {accuracy}, confusion matrix: \n{conf_matrix}")


    def train(self, train_ts, test_ts, S, Q, learning_rate, iters):
        step_size = 25

        nll_train_arr, nll_test_arr = np.zeros(iters), np.zeros(iters)
        acc_arr = np.zeros(math.ceil(iters/step_size))

        # Randomly initialise random student, question parameters
        bs = torch.randn(S, requires_grad=True, generator=self.rng)
        bq = torch.randn(Q, requires_grad=True, generator=self.rng)

        for epoch in range(iters):
            # Train set params
            bs_train = torch.index_select(bs, 0, train_ts[1])
            bq_train = torch.index_select(bq, 0, train_ts[2])
            # Train nll
            probit_1 = 1/(1+torch.exp(-bs_train-bq_train))
            nll = -torch.sum(train_ts[0]*torch.log(probit_1) + (1-train_ts[0])*torch.log(1-probit_1))
            nll.backward()

            # Test set params
            bs_copy, bq_copy = torch.clone(bs), torch.clone(bq)
            bs_test = torch.index_select(bs_copy, 0, test_ts[1])
            bq_test = torch.index_select(bq_copy, 0, test_ts[2])
            # Test nll
            probit_1_test = 1/(1+torch.exp(-bs_test-bq_test))
            nll_test = -torch.sum(test_ts[0]*torch.log(probit_1_test) + (1-test_ts[0])*torch.log(1-probit_1_test))

            # Gradient descent
            with torch.no_grad():
                bs -= learning_rate * bs.grad
                bq -= learning_rate * bq.grad

            # Zero gradients after updating
            bs.grad.zero_()
            bq.grad.zero_()

            if epoch % step_size == 0:
                acc, _ = self.predict(bs, bq, test_ts)
                acc_arr[epoch // step_size] = acc
                print(epoch, nll, nll_test, acc)

            nll_train_arr[epoch], nll_test_arr[epoch] = nll, nll_test

        print(epoch, nll, nll_test, acc)
        return bs, bq, nll_train_arr, nll_test_arr, acc_arr


    def predict(self, bs, bq, test_ts, vis=False):
        # Test set params after training
        bs_test = torch.index_select(bs, 0, test_ts[1])
        bq_test = torch.index_select(bq, 0, test_ts[2])

        predicted_probit = ADP.probit_correct(bs_test, bq_test)
        predictions = torch.bernoulli(predicted_probit, generator=self.rng)

        performance = torch.sum(torch.eq(test_ts[0], predictions)) / torch.numel(test_ts[0])
        performance = float(performance)*100

        conf_matrix = confusion_matrix(test_ts[0].numpy(), predictions.detach().numpy())
        conf_matrix = conf_matrix*100/torch.numel(test_ts[0])
        
        if vis:
            # Visualisation (only possible under certain conditions)
            no_questions = self.testset_col_range[1] - self.testset_col_range[0]
            test_ts_reshaped = test_ts[0].reshape(int(len(test_ts[0])/no_questions),no_questions)
            predicted_probit_reshaped = predicted_probit.reshape(int(len(predicted_probit)/no_questions),no_questions)
            predictions_reshaped = predictions.reshape(int(len(predictions)/no_questions),no_questions)
            
            real_portion = test_ts_reshaped.detach()
            real_portion = real_portion[:50, :]
            sns.heatmap(real_portion, linewidth=0.5)
            plt.title('Real binarised data')
            plt.xlabel('Questions')
            plt.ylabel('Students')
            plt.show()

            predicted_probit_portion = predicted_probit_reshaped.detach()
            predicted_probit_portion = predicted_probit_portion[:50, :]
            sns.heatmap(predicted_probit_portion, linewidth=0.5)
            plt.title('Predicted probabilities')
            plt.xlabel('Questions')
            plt.ylabel('Students')
            plt.show()

            predicted_portion = predictions_reshaped.detach()
            predicted_portion = predicted_portion[:50, :]
            sns.heatmap(predicted_portion, linewidth=0.5)
            plt.title('Predicted output')
            plt.xlabel('Questions')
            plt.ylabel('Students')
            plt.show()

        return performance, conf_matrix


    def plot_result(self, avg_nll_train_arr, avg_nll_test_arr, acc_arr, iters):
        step_size = 25
        plt.plot(range(iters), avg_nll_train_arr)
        plt.plot(range(iters), avg_nll_test_arr)
        plt.plot(np.arange(0, iters, step_size), acc_arr/100)
        plt.title('Train and test nll')
        plt.ylabel('Negative log likelihood')
        plt.xlabel('epoch')
        plt.legend(['Train nll', 'Test nll', 'Accuracy'])
        plt.show()
