import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from utils.vectorise_data import vectorise_data
from utils.split_data import split_to_4quadrants, split_to_4quadrants_df

def probit_correct(bs, bq, ps):
    return 1/(1+torch.exp(-bs-bq-ps))

def train(train_ts, test_ts, meta_ts, S, Q, rng, learning_rate, iters):

    nll_train_arr, nll_test_arr = np.zeros(iters), np.zeros(iters)

    # Randomly initialise student, question, problem solving parameters
    bs = torch.randn(S, requires_grad=True, generator=rng)
    bq = torch.randn(Q, requires_grad=True, generator=rng)
    ws = torch.randn(S, requires_grad=True, generator=rng)

    for epoch in range(iters):
        # Train set params
        bs_train = torch.index_select(bs, 0, train_ts[1])
        bq_train = torch.index_select(bq, 0, train_ts[2])
        ws_train = torch.index_select(ws, 0, train_ts[1])
        # Train nll
        meta_train = torch.index_select(meta_ts, 0, train_ts[2])
        ps_train = ws_train*meta_train
        probit_1 = 1/(1+torch.exp(-bs_train-bq_train-ps_train))
        nll = -torch.sum(train_ts[0]*torch.log(probit_1) + (1-train_ts[0])*torch.log(1-probit_1))
        nll.backward()

        # Test set params
        bs_test = torch.index_select(bs, 0, test_ts[1])
        bq_test = torch.index_select(bq, 0, test_ts[2])
        ws_test = torch.index_select(ws, 0, test_ts[1])
        # Test nll
        nll_test = 0
        meta_test = torch.index_select(meta_ts, 0, test_ts[2])
        ps_test = ws_test*meta_test
        probit_1_test = 1/(1+torch.exp(-bs_test-bq_test-ps_test))
        nll_test = -torch.sum(test_ts[0]*torch.log(probit_1_test) + (1-test_ts[0])*torch.log(1-probit_1_test))

        # Gradient descent
        with torch.no_grad():
            bs -= learning_rate * bs.grad
            bq -= learning_rate * bq.grad
            ws -= learning_rate * ws.grad

        # Zero gradients after updating
        bs.grad.zero_()
        bq.grad.zero_()
        ws.grad.zero_()

        if epoch % 10 == 0:
            print(epoch, nll, nll_test)
        
        nll_train_arr[epoch], nll_test_arr[epoch] = nll, nll_test

    print(epoch, nll, nll_test)
    return bs, bq, ws, nll_train_arr, nll_test_arr


def predict(trained_bs, trained_bq, trained_ws, test_ts, meta_ts, rng, Q_test):
    
    # Test set params after training
    bs_test = torch.index_select(trained_bs, 0, test_ts[1])
    bq_test = torch.index_select(trained_bq, 0, test_ts[2])
    ws_test = torch.index_select(trained_ws, 0, test_ts[1])

    meta_test = torch.index_select(meta_ts, 0, test_ts[2])
    ps_test = ws_test*meta_test

    predicted_probit = probit_correct(bs_test, bq_test, ps_test)
    predictions = torch.bernoulli(predicted_probit, generator=rng)

    performance = torch.sum(torch.eq(test_ts[0], predictions)) / torch.numel(test_ts[0])
    performance = float(performance)*100

    conf_matrix = confusion_matrix(test_ts[0].numpy(), predictions.detach().numpy())
    conf_matrix = conf_matrix*100/torch.numel(test_ts[0])
    
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


def train_meta_vectorised(train_vectorised_ts, test_vectorised_ts, S, Q, meta_ts, rng, learning_rate, iters, Q_test):

    trained_bs, trained_bq, trained_ws, nll_train_arr, nll_test_arr = train(train_vectorised_ts, test_vectorised_ts, meta_ts, S, Q, rng, learning_rate, iters)
    train_ts_size, test_ts_size = train_vectorised_ts.shape[1], test_vectorised_ts.shape[1]

    plt.plot(range(iters), nll_train_arr/train_ts_size)
    plt.plot(range(iters), nll_test_arr/test_ts_size)
    plt.title('Train and test nll')
    plt.ylabel('Negative log likelihood')
    plt.xlabel('epoch')
    plt.legend(['Train nll', 'Test nll'])
    plt.show()

    performance, conf_matrix = predict(trained_bs, trained_bq, trained_ws, test_vectorised_ts, meta_ts, rng, Q_test)
    return performance, conf_matrix
