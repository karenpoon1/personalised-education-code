import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from models.ADP_quadrant import probit_correct

def train(train_ts, test_ts, S, Q, rng, learning_rate, iters):

    nll_train_arr, nll_test_arr = np.zeros(iters), np.zeros(iters)
    acc_arr, epoch_arr = [], []

    # Randomly initialise random student, question parameters
    bs = torch.randn(S, requires_grad=True, generator=rng)
    bq = torch.randn(Q, requires_grad=True, generator=rng)

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

        if epoch % 25 == 0:
            acc, _ = predict(bs, bq, test_ts, rng)
            epoch_arr.append(epoch)
            acc_arr.append(acc)
            print(epoch, nll, nll_test, acc)

        nll_train_arr[epoch], nll_test_arr[epoch] = nll, nll_test

    print(epoch, nll, nll_test)
    return bs, bq, nll_train_arr, nll_test_arr, epoch_arr, acc_arr


def predict(bs, bq, test_ts, rng, Q_test=None, vis=False):
    
    # Test set params after training
    bs_test = torch.index_select(bs, 0, test_ts[1])
    bq_test = torch.index_select(bq, 0, test_ts[2])

    predicted_probit = probit_correct(bs_test, bq_test)
    predictions = torch.bernoulli(predicted_probit, generator=rng)

    performance = torch.sum(torch.eq(test_ts[0], predictions)) / torch.numel(test_ts[0])
    performance = float(performance)*100

    conf_matrix = confusion_matrix(test_ts[0].numpy(), predictions.detach().numpy())
    conf_matrix = conf_matrix*100/torch.numel(test_ts[0])
    
    if vis:
        # Visualisation
        no_questions = Q_test
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


def train_product_vectorised(train_ts, test_ts, S, Q, rng, learning_rate, iters, Q_test):

    bs, bq, nll_train_arr, nll_test_arr, epoch_arr, acc_arr = train(train_ts, test_ts, S, Q, rng, learning_rate, iters)
    train_ts_size, test_ts_size = train_ts.shape[1], test_ts.shape[1]

    plt.plot(range(iters), nll_train_arr/train_ts_size)
    plt.plot(range(iters), nll_test_arr/test_ts_size)
    plt.plot(np.array(epoch_arr), np.array(acc_arr)/100)
    plt.title('Train and test nll')
    plt.ylabel('Negative log likelihood')
    plt.xlabel('epoch')
    plt.legend(['Train nll', 'Test nll', 'Accuracy'])
    plt.show()

    performance, conf_matrix = predict(bs, bq, test_ts, rng, Q_test, vis=True)
    return performance, conf_matrix
