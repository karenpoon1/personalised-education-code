import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from utils.vectorise_data import vectorise_data
from models.ability_difficulty_product import probit_correct

def train(train_ts_vectorised, test_ts_vectorised, S, Q, rng, learning_rate, n_iters):

    nll_train_arr, nll_test_arr = np.zeros(n_iters), np.zeros(n_iters)

    bs = torch.randn(S, requires_grad=True, generator=rng)
    bq = torch.randn(Q, requires_grad=True, generator=rng)

    for epoch in range(n_iters):
        bs_vector = torch.index_select(bs, 0, train_ts_vectorised[1])
        bq_vector = torch.index_select(bq, 0, train_ts_vectorised[2])

        probit_1 = 1/(1+torch.exp(-bs_vector-bq_vector))
        nll = -torch.sum(train_ts_vectorised[0]*torch.log(probit_1) + (1-train_ts_vectorised[0])*torch.log(1-probit_1))
        nll.backward()

        # calc test nll
        bs_vector_test = torch.index_select(bs, 0, test_ts_vectorised[1])
        bq_vector_test = torch.index_select(bq, 0, test_ts_vectorised[2])

        probit_1_test = 1/(1+torch.exp(-bs_vector_test-bq_vector_test))
        nll_test = -torch.sum(test_ts_vectorised[0]*torch.log(probit_1_test) + (1-test_ts_vectorised[0])*torch.log(1-probit_1_test))

        with torch.no_grad():
            bs -= learning_rate * bs.grad
            bq -= learning_rate * bq.grad

        # zero the gradients after updating
        bs.grad.zero_()
        bq.grad.zero_()

        if epoch % 100 == 0:
            print(epoch, nll, nll_test)

        nll_train_arr[epoch], nll_test_arr[epoch] = nll, nll_test

    return bs, bq, n_iters, nll_train_arr, nll_test_arr


def predict(bs, bq, test_ts, rng):
    
    bs_vector = torch.index_select(bs, 0, test_ts[1])
    bq_vector = torch.index_select(bq, 0, test_ts[2])

    product_params_matrix = probit_correct(bs_vector, bq_vector)

    predictions = torch.bernoulli(product_params_matrix, generator=rng)

    performance = torch.sum(torch.eq(test_ts[0], predictions)) / torch.numel(test_ts[0])
    performance = float(performance)*100

    conf_matrix = confusion_matrix(test_ts[0].numpy(), predictions.detach().numpy())
    conf_matrix = conf_matrix*100/torch.numel(test_ts[0])
    
    # real_portion = test_ts.detach()
    # real_portion = real_portion[:50, :]
    # sns.heatmap(real_portion, linewidth=0.5)
    # plt.title('Real binarised data')
    # plt.show()

    # portion = product_params_matrix.detach()
    # portion = portion[:50, :]
    # sns.heatmap(portion, linewidth=0.5)
    # plt.title('Predicted probabilities')
    # plt.show()
    return product_params_matrix, performance, conf_matrix


def train_product_vectorised(ts, df, rng, learning_rate, n_iters):

    # Retaining pervious test set:
    # first_quadrant_vectorised_ts = vectorise_data(first_quadrant_ts, first_quadrant_df)
    # train_question_vectorised_ts = vectorise_data(train_question_ts, train_question_df)
    # train_student_vectorised_ts = vectorise_data(train_student_ts, train_student_df)
    # train_vectorised_ts = torch.cat((first_quadrant_vectorised_ts, train_question_vectorised_ts, train_student_vectorised_ts), dim=1)

    S, Q = ts.shape[0], ts.shape[1]
    vectorised_ts = vectorise_data(ts, df)

    # shuffle
    col_idxs = list(range(vectorised_ts.shape[1]))
    random.seed(1000)
    random.shuffle(col_idxs)
    shuffled_ts = vectorised_ts[:, torch.tensor(col_idxs)]
    
    train_ts, test_ts = torch.split(shuffled_ts, int(S*Q*3/4), dim=1)
    train_ts_size, test_ts_size = train_ts.shape[1], test_ts.shape[1]

    bs, bq, n_iters, nll_train_arr, nll_test_arr = train(train_ts, test_ts, S, Q, rng, learning_rate, n_iters)
    plt.plot(range(n_iters), nll_train_arr/train_ts_size)
    plt.plot(range(n_iters), nll_test_arr/test_ts_size)
    plt.title('Train and test nll')
    plt.ylabel('Negative log likelihood')
    plt.xlabel('epoch')
    plt.legend(['Train nll', 'Test nll'])
    plt.show()

    product_params_matrix, performance, conf_matrix = predict(bs, bq, test_ts, rng)

    print(f"bs (student params): {bs}")
    print(f"bq (question params): {bq}")
    print(f"Predicted probabilities: {product_params_matrix}")
    print(f"Percentage accuracy for product baseline: {performance}")
    print(f"Confusion matrix: {conf_matrix}")

    return bs, bq, product_params_matrix, performance, conf_matrix
