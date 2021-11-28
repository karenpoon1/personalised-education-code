import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from utils.vectorise_data import vectorise_data
from utils.split_data import split_to_4quadrants, split_to_4quadrants_df

def probit_correct(bs, bq, i):
    return 1/(1+torch.exp(-bs-bq-i))

def train(train_ts_vectorised, test_ts_vectorised, meta_ts, S, Q, rng, learning_rate, n_iters):

    nll_train_arr, nll_test_arr = np.zeros(n_iters), np.zeros(n_iters)
    
    dimension = 3
    bs = torch.randn(S, requires_grad=True, generator=rng)
    bq = torch.randn(Q, requires_grad=True, generator=rng)
    xs = torch.randn((dimension,S), requires_grad=True, generator=rng)
    xq = torch.randn((dimension,Q), requires_grad=True, generator=rng)

    for epoch in range(n_iters):
        bs_vector = torch.index_select(bs, 0, train_ts_vectorised[1])
        bq_vector = torch.index_select(bq, 0, train_ts_vectorised[2])

        xs_vector = torch.index_select(xs, 1, train_ts_vectorised[1])
        xq_vector = torch.index_select(xq, 1, train_ts_vectorised[2])
        
        # interactive bias
        interactive_vector = xs_vector*xq_vector
        interactive_vector = torch.sum(interactive_vector, 0)

        probit_1 = 1/(1+torch.exp(-bs_vector-bq_vector-interactive_vector))
        nll = -torch.sum(train_ts_vectorised[0]*torch.log(probit_1) + (1-train_ts_vectorised[0])*torch.log(1-probit_1))
        nll.backward()

        # calc test nll
        nll_test = 0
        bs_vector_test = torch.index_select(bs, 0, test_ts_vectorised[1])
        bq_vector_test = torch.index_select(bq, 0, test_ts_vectorised[2])

        xs_vector_test = torch.index_select(xs, 1, test_ts_vectorised[1])
        xq_vector_test = torch.index_select(xq, 1, test_ts_vectorised[2])

        interactive_vector_test = xs_vector_test*xq_vector_test
        interactive_vector_test = torch.sum(interactive_vector_test, 0)

        probit_1_test = 1/(1+torch.exp(-bs_vector_test-bq_vector_test-interactive_vector_test))
        nll_test = -torch.sum(test_ts_vectorised[0]*torch.log(probit_1_test) + (1-test_ts_vectorised[0])*torch.log(1-probit_1_test))

        with torch.no_grad():
            bs -= learning_rate * bs.grad
            bq -= learning_rate * bq.grad
            xs -= learning_rate * xs.grad
            xq -= learning_rate * xq.grad

        # zero the gradients after updating
        bs.grad.zero_()
        bq.grad.zero_()
        xs.grad.zero_()
        xq.grad.zero_()

        if epoch % 100 == 0:
            print(epoch, nll, nll_test)
        
        nll_train_arr[epoch], nll_test_arr[epoch] = nll, nll_test

    return bs, bq, xs, xq, n_iters, nll_train_arr, nll_test_arr


def predict(bs, bq, xs, xq, test_ts, meta_ts, rng, held_out_test):
    
    bs_vector = torch.index_select(bs, 0, test_ts[1])
    bq_vector = torch.index_select(bq, 0, test_ts[2])

    xs_vector = torch.index_select(xs, 1, test_ts[1])
    xq_vector = torch.index_select(xq, 1, test_ts[2])

    interactive_vector = xs_vector*xq_vector
    interactive_vector = torch.sum(interactive_vector, 0)

    product_params_matrix = probit_correct(bs_vector, bq_vector, interactive_vector)

    predictions = torch.bernoulli(product_params_matrix, generator=rng)

    performance = torch.sum(torch.eq(test_ts[0], predictions)) / torch.numel(test_ts[0])
    performance = float(performance)*100

    conf_matrix = confusion_matrix(test_ts[0].numpy(), predictions.detach().numpy())
    conf_matrix = conf_matrix*100/torch.numel(test_ts[0])
    
    if held_out_test == True:
        no_questions = 12
        print(test_ts[0], len(test_ts[0]))
        test_ts_reshaped = test_ts[0].reshape(int(len(test_ts[0])/no_questions),no_questions)
        product_params_matrix_reshaped = product_params_matrix.reshape(int(len(product_params_matrix)/no_questions),no_questions)
        predictions_reshaped = predictions.reshape(int(len(predictions)/no_questions),no_questions)
        
        real_portion = test_ts_reshaped.detach()
        real_portion = real_portion[:50, :]
        sns.heatmap(real_portion, linewidth=0.5)
        plt.title('Real binarised data')
        plt.xlabel('Questions')
        plt.ylabel('Students')
        plt.show()

        predicted_probit_portion = product_params_matrix_reshaped.detach()
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

    return product_params_matrix, performance, conf_matrix


def train_product_interactive(ts, df, meta_ts, rng, learning_rate, n_iters, held_out_test):

    S, Q = ts.shape[0], ts.shape[1]

    if held_out_test == True:
        # Retaining pervious test set:
        first_quadrant_ts, train_question_ts, train_student_ts, test_ts = split_to_4quadrants(ts)
        first_quadrant_df, train_question_df, train_student_df, test_df = split_to_4quadrants_df(df)

        first_quadrant_vectorised_ts = vectorise_data(first_quadrant_ts, first_quadrant_df)
        train_question_vectorised_ts = vectorise_data(train_question_ts, train_question_df)
        train_student_vectorised_ts = vectorise_data(train_student_ts, train_student_df)

        train_vectorised_ts = torch.cat((first_quadrant_vectorised_ts, train_question_vectorised_ts, train_student_vectorised_ts), dim=1)
        test_vectorised_ts = vectorise_data(test_ts, test_df)

    else:
        vectorised_ts = vectorise_data(ts, df)

        # shuffle
        col_idxs = list(range(vectorised_ts.shape[1]))
        random.seed(1000)
        random.shuffle(col_idxs)
        shuffled_ts = vectorised_ts[:, torch.tensor(col_idxs)]
        
        train_vectorised_ts, test_vectorised_ts = torch.split(shuffled_ts, int(S*Q*3/4), dim=1)

    train_ts_size, test_ts_size = train_vectorised_ts.shape[1], test_vectorised_ts.shape[1]

    bs, bq, xs, xq, n_iters, nll_train_arr, nll_test_arr = train(train_vectorised_ts, test_vectorised_ts, meta_ts, S, Q, rng, learning_rate, n_iters)
    plt.plot(range(n_iters), nll_train_arr/train_ts_size)
    plt.plot(range(n_iters), nll_test_arr/test_ts_size)
    plt.title('Train and test nll')
    plt.ylabel('Negative log likelihood')
    plt.xlabel('epoch')
    plt.legend(['Train nll', 'Test nll'])
    plt.show()

    product_params_matrix, performance, conf_matrix = predict(bs, bq, xs, xq, test_vectorised_ts, meta_ts, rng, held_out_test)

    print(f"bs (student params): {bs}")
    print(f"bq (question params): {bq}")
    print(f"xs (interative student multi dimensional params): {xs}")
    print(f"xq (interative question multi dimensional params): {xq}")
    print(f"Predicted probabilities: {product_params_matrix}")
    print(f"Percentage accuracy for product baseline: {performance}")
    print(f"Confusion matrix: {conf_matrix}")

    return bs, bq, product_params_matrix, performance, conf_matrix
