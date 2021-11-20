import torch
import random
import matplotlib.pyplot as plt
import seaborn as sns
from utils.shuffle_data import shuffle_cols
from models.ability_difficulty_product import probit_correct

def train(learning_rate, n_iters, train_ts_vectorised, test_ts_vectorised, S, Q, rng):

    t_arr, nll_train_arr, nll_test_arr = [], [], []

    bs = torch.randn(S, requires_grad=True, generator=rng)
    bq = torch.randn(Q, requires_grad=True, generator=rng)

    for epoch in range(n_iters):
        bs_vector = torch.index_select(bs, 0, train_ts_vectorised[1])
        bq_vector = torch.index_select(bq, 0, train_ts_vectorised[2])

        probit_1 = 1/(1+torch.exp(-bs_vector-bq_vector))
        probit_0 = 1 - probit_1
        nll = -torch.sum(train_ts_vectorised[0]*torch.log(probit_1) + (1-train_ts_vectorised[0])*torch.log(probit_0))
        nll.backward()

        # calc test nll
        bs_vector = torch.index_select(bs, 0, test_ts_vectorised[1])
        bq_vector = torch.index_select(bq, 0, test_ts_vectorised[2])

        probit_1 = 1/(1+torch.exp(-bs_vector-bq_vector))
        probit_0 = 1 - probit_1
        nll_test = -torch.sum(test_ts_vectorised[0]*torch.log(probit_1) + (1-test_ts_vectorised[0])*torch.log(probit_0))

        with torch.no_grad():
            bs -= learning_rate * bs.grad
            bq -= learning_rate * bq.grad

        # zero the gradients after updating
        bs.grad.zero_()
        bq.grad.zero_()

        if epoch % 100 == 0:
            print(epoch, nll, nll_test)

        t_arr.append(epoch)
        nll_train_arr.append(nll)
        nll_test_arr.append(nll_test)

    return bs, bq, t_arr, nll_train_arr, nll_test_arr


def predict(bs, bq, test_ts, rng):
    
    bs_vector = torch.index_select(bs, 0, test_ts[1])
    bq_vector = torch.index_select(bq, 0, test_ts[2])

    product_params_matrix = probit_correct(bs_vector, bq_vector)

    predictions = torch.bernoulli(product_params_matrix, generator=rng)

    performance = torch.sum(torch.eq(test_ts[0], predictions)) / torch.numel(test_ts[0])
    performance = float(performance)*100
    
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
    return product_params_matrix, performance


def train_product_vectorised(ts, df, S, Q, learning_rate, n_iters, rng):
    # first_quadrant_vectorised_ts = vectorise_data(first_quadrant_ts, first_quadrant_df)
    # train_question_vectorised_ts = vectorise_data(train_question_ts, train_question_df)
    # train_student_vectorised_ts = vectorise_data(train_student_ts, train_student_df)
    # train_vectorised_ts = torch.cat((first_quadrant_vectorised_ts, train_question_vectorised_ts, train_student_vectorised_ts), dim=1)
    vectorised_ts = vectorise_data(ts, df)

    # shuffle
    col_idxs = list(range(vectorised_ts.shape[1]))
    random.seed(1000)
    random.shuffle(col_idxs)
    shuffled_ts = vectorised_ts[:, torch.tensor(col_idxs)]
    
    train_ts, test_ts = torch.split(shuffled_ts, int(S*Q*3/4), dim=1)

    bs, bq, t_arr, nll_train_arr, nll_test_arr = train(learning_rate, n_iters, train_ts, test_ts, S, Q, rng)
    plt.plot(t_arr, nll_train_arr)
    plt.title('Train and test nll')
    plt.ylabel('Negative log likelihood')
    plt.xlabel('epoch')

    plt.plot(t_arr, nll_test_arr)
    plt.legend(['Train nll', 'Test nll'])
    plt.show()

    product_params_matrix, performance = predict(bs, bq, test_ts, rng)

    print(f"bs (student params): {bs}")
    print(f"bq (question params): {bq}")
    print(f"Predicted probabilities: {product_params_matrix}")
    print(f"Percentage accuracy for product baseline: {performance}")

    return bs, bq, product_params_matrix, performance

def vectorise_data(data_ts, data_df):
    S, Q = data_ts.shape[0], data_ts.shape[1]
    
    reshaped_data = data_ts.reshape(-1).type(torch.int) # unstack data
    
    student_id = torch.tensor(data_df.index.values)
    student_id = student_id.repeat(Q, 1).T.reshape(-1)
    
    question_id = torch.tensor([int(entry[1:])-1 for entry in data_df.columns.tolist()])
    question_id = question_id.repeat(S)

    vectorised_data_ts = torch.stack((reshaped_data, student_id, question_id), dim=0)

    return vectorised_data_ts
