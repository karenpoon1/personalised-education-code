import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from utils.split_and_vectorise import split_and_vectorise

class I:
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
    def probit_correct(bs, bq, i):
        return 1/(1+torch.exp(-bs-bq-i))

    def run(self, learning_rate, iters, dimension=1, validation=0.9):
        train_vectorised_ts, test_vectorised_ts = split_and_vectorise(self.data_ts, self.testset_row_range, self.testset_col_range, shuffle=True)
        train_vectorised_ts, validation_vectorised_ts = torch.split(train_vectorised_ts, int(torch.numel(train_vectorised_ts[0])*validation), dim=1)

        S, Q = self.data_ts.shape[0], self.data_ts.shape[1] # Data block size
        trained_bs, trained_bq, trained_xs, trained_xq, nll_train_arr, nll_test_arr, nll_validation_arr, train_acc_arr, test_acc_arr = self.train(train_vectorised_ts, validation_vectorised_ts, test_vectorised_ts, S, Q, learning_rate, iters, dimension)
        accuracy, conf_matrix = self.predict(trained_bs, trained_bq, trained_xs, trained_xq, test_vectorised_ts, vis=True)

        self.trained_bs = trained_bs
        self.trained_bq = trained_bq
        self.trained_xs = trained_xs
        self.trained_xq = trained_xq
        self.nll_train_arr = nll_train_arr
        self.nll_test_arr = nll_test_arr
        self.nll_validation_arr = nll_validation_arr
        self.train_acc_arr = train_acc_arr
        self.test_acc_arr = test_acc_arr
        self.accuracy = accuracy
        self.conf_matrix = conf_matrix

        self.plot_result(nll_train_arr/train_vectorised_ts.shape[1], nll_test_arr/test_vectorised_ts.shape[1], nll_validation_arr/validation_vectorised_ts.shape[1], train_acc_arr, test_acc_arr, iters)
        print(f"Interactive (rate={learning_rate}, iters={iters}) -> accuracy: {accuracy}, confusion matrix: \n{conf_matrix}")


    def train(self, train_ts, validation_ts, test_ts, S, Q, learning_rate, iters, dimension):
        step_size = 25

        nll_train_arr, nll_test_arr, nll_validation_arr = np.zeros(iters), np.zeros(iters), np.zeros(iters)
        train_acc_arr = np.zeros(math.ceil(iters/step_size))
        test_acc_arr = np.zeros(math.ceil(iters/step_size))

        # Randomly initialise random student, question parameters
        bs = torch.randn(S, requires_grad=True, generator=self.rng)
        bq = torch.randn(Q, requires_grad=True, generator=self.rng)
        xs = torch.randn((dimension,S), requires_grad=True, generator=self.rng)
        xq = torch.randn((dimension,Q), requires_grad=True, generator=self.rng)

        for epoch in range(iters):
            # Train set params
            bs_train = torch.index_select(bs, 0, train_ts[1])
            bq_train = torch.index_select(bq, 0, train_ts[2])
            xs_train = torch.index_select(xs, 1, train_ts[1])
            xq_train = torch.index_select(xq, 1, train_ts[2])
            # Train nll
            interactive_train = xs_train*xq_train
            interactive_train = torch.sum(interactive_train, 0)
            probit_1 = 1/(1+torch.exp(-bs_train-bq_train-interactive_train))
            nll = -torch.sum(train_ts[0]*torch.log(probit_1) + (1-train_ts[0])*torch.log(1-probit_1))
            nll.backward()
            
            # test set params
            bs_copy, bq_copy, xs_copy, xq_copy = torch.clone(bs), torch.clone(bq), torch.clone(xs), torch.clone(xq)
            bs_test = torch.index_select(bs_copy, 0, test_ts[1])
            bq_test = torch.index_select(bq_copy, 0, test_ts[2])
            xs_test = torch.index_select(xs_copy, 1, test_ts[1])
            xq_test = torch.index_select(xq_copy, 1, test_ts[2])
            # test nll
            nll_test = 0
            interactive_test = xs_test*xq_test
            interactive_test = torch.sum(interactive_test, 0)
            probit_1_test = 1/(1+torch.exp(-bs_test-bq_test-interactive_test))
            nll_test = -torch.sum(test_ts[0]*torch.log(probit_1_test) + (1-test_ts[0])*torch.log(1-probit_1_test))

            # validation set params
            bs_copy, bq_copy, xs_copy, xq_copy = torch.clone(bs), torch.clone(bq), torch.clone(xs), torch.clone(xq)
            bs_validation = torch.index_select(bs_copy, 0, validation_ts[1])
            bq_validation = torch.index_select(bq_copy, 0, validation_ts[2])
            xs_validation = torch.index_select(xs_copy, 1, validation_ts[1])
            xq_validation = torch.index_select(xq_copy, 1, validation_ts[2])
            # validation nll
            nll_validation = 0
            interactive_validation = xs_validation*xq_validation
            interactive_validation = torch.sum(interactive_validation, 0)
            probit_1_validation = 1/(1+torch.exp(-bs_validation-bq_validation-interactive_validation))
            nll_validation = -torch.sum(validation_ts[0]*torch.log(probit_1_validation) + (1-validation_ts[0])*torch.log(1-probit_1_validation))
            
            # Gradient descent
            with torch.no_grad():
                bs -= learning_rate * bs.grad
                bq -= learning_rate * bq.grad
                xs -= learning_rate * xs.grad
                xq -= learning_rate * xq.grad

            # Zero gradients after updating
            bs.grad.zero_()
            bq.grad.zero_()
            xs.grad.zero_()
            xq.grad.zero_()

            if epoch % step_size == 0:
                test_acc, _ = self.predict(bs, bq, xs, xq, test_ts)
                test_acc_arr[epoch // step_size] = test_acc

                train_acc, _ = self.predict(bs, bq, xs, xq, train_ts)
                train_acc_arr[epoch // step_size] = train_acc

                print(epoch, nll, nll_validation, train_acc, test_acc)

            # if epoch > 2325:
            #     print(epoch, nll, nll_validation, test_acc)
            #     print(bs, bq, xs, xq)

            nll_train_arr[epoch], nll_test_arr[epoch], nll_validation_arr[epoch] = nll, nll_test, nll_validation

        print(epoch, nll, nll_validation, test_acc)
        return bs, bq, xs, xq, nll_train_arr, nll_test_arr, nll_validation_arr, train_acc_arr, test_acc_arr


    def predict(self, trained_bs, trained_bq, trained_xs, trained_xq, test_ts, vis=False):
        # Test set params after training

        bs_test = torch.index_select(trained_bs, 0, test_ts[1])
        bq_test = torch.index_select(trained_bq, 0, test_ts[2])
        xs_test = torch.index_select(trained_xs, 1, test_ts[1])
        xq_test = torch.index_select(trained_xq, 1, test_ts[2])

        interactive_test = xs_test*xq_test
        interactive_test = torch.sum(interactive_test, 0)
        predicted_probit = I.probit_correct(bs_test, bq_test, interactive_test)
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
            real_portion = real_portion[:30, :]
            sns.heatmap(real_portion, linewidth=0.5)
            plt.title('Real binarised data')
            plt.xlabel('Questions')
            plt.ylabel('Students')
            plt.show()

            predicted_probit_portion = predicted_probit_reshaped.detach()
            predicted_probit_portion = predicted_probit_portion[:30, :]
            sns.heatmap(predicted_probit_portion, linewidth=0.5)
            plt.title('Predicted probabilities')
            plt.xlabel('Questions')
            plt.ylabel('Students')
            plt.show()

            predicted_portion = predictions_reshaped.detach()
            predicted_portion = predicted_portion[:30, :]
            sns.heatmap(predicted_portion, linewidth=0.5)
            plt.title('Predicted output')
            plt.xlabel('Questions')
            plt.ylabel('Students')
            plt.show()

        return performance, conf_matrix


    def plot_result(self, avg_nll_train_arr, avg_nll_test_arr, avg_nll_validation_arr, train_acc_arr, test_acc_arr, iters):
        step_size = 25
        plt.plot(range(iters), avg_nll_train_arr)
        plt.plot(range(iters), avg_nll_validation_arr)
        plt.plot(range(iters), avg_nll_test_arr)
        plt.plot(np.arange(0, iters, step_size), train_acc_arr/100)
        plt.plot(np.arange(0, iters, step_size), test_acc_arr/100)
        plt.title('Train and test nll')
        plt.ylabel('Negative log likelihood')
        plt.xlabel('epoch')
        plt.legend(['Train nll', 'Test nll', 'Validation nll', 'Train accuracy', 'Test accuracy'])
        plt.show()
