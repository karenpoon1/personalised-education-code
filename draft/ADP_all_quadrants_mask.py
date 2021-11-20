from numpy.core.numeric import full
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.clean_data import thres_score_range
from utils.binarise_data import binarise_by_mid, binarise_by_avg
from utils.shuffle_data import shuffle_cols
from utils.split_data import split_to_4quadrants

df = pd.read_csv("Fwd__Pinpoint_ML_Dataset/9to1_2017_GCSE_1H.csv", skiprows=list(range(1,24)), usecols=list(range(2,26)))
max_scores = pd.read_csv("Fwd__Pinpoint_ML_Dataset/9to1_2017_GCSE_1H.csv", nrows = 1, skiprows=[1], usecols=list(range(2,26)))

def train(learning_rate, n_iters, dataset_ts, mask, rng):

    t_arr, nll_arr = [], []
    S = dataset_ts.shape[0] # no. of rows
    Q = dataset_ts.shape[1] # no. of cols

    bs_tensor = torch.randn(S, requires_grad=True, generator=rng)
    bq_tensor = torch.randn(Q, requires_grad=True, generator=rng)
    
    # bs_tensor = Variable(torch.normal(mean=0, std=np.sqrt(10), size=(S,), generator=rng), requires_grad=True)
    # bq_tensor = Variable(torch.normal(mean=0, std=np.sqrt(10), size=(Q,), generator=rng), requires_grad=True)

    for epoch in range(n_iters):

        bs_matrix = bs_tensor.repeat(Q, 1)
        bs_matrix = torch.transpose(bs_matrix, 0, 1)
        bq_matrix = bq_tensor.repeat(S, 1)

        probit_1 = torch.log(1/(1+torch.exp(-bs_matrix-bq_matrix)))
        probit_0 = torch.log(1/(1+torch.exp(+bs_matrix+bq_matrix)))
        nll_matrix = dataset_ts*probit_1 + (1-dataset_ts)*probit_0
        masked_nll_matrix = nll_matrix * mask
        nll = -torch.sum(masked_nll_matrix)
        # nll = -torch.nansum(dataset_ts*probit_1 + (1-dataset_ts)*probit_0)
        # nll = -torch.sum(dataset_ts*probit_1 + (1-dataset_ts)*probit_0)

        nll.backward()
        # print(nll)

        with torch.no_grad():
            bs_tensor -= learning_rate * bs_tensor.grad
            # print(bs_tensor.grad)
            bq_tensor -= learning_rate * bq_tensor.grad
            # print(bq_tensor.grad)

        # zero the gradients after updating
        bs_tensor.grad.zero_()
        bq_tensor.grad.zero_()

        if epoch % 100 == 0:
            print(epoch,nll)

        t_arr.append(epoch)
        nll_arr.append(nll)

    return bs_tensor, bq_tensor, t_arr, nll_arr

def probit_correct(bs, bq):
    return 1/(1+torch.exp(-bs-bq))

def predict(bs_tensor, bq_tensor, test_output_ts, rng):
    bs_matrix = bs_tensor.repeat(len(bq_tensor), 1)
    bs_matrix = torch.transpose(bs_matrix, 0, 1)
    bq_matrix = bq_tensor.repeat(len(bs_tensor), 1)

    product_params_matrix = probit_correct(bs_matrix, bq_matrix)

    predictions = torch.bernoulli(product_params_matrix, generator=rng)

    performance = torch.sum(torch.eq(test_output_ts, predictions)) / torch.numel(test_output_ts)
    performance = float(performance)*100

    real_portion = test_output_ts.detach()
    real_portion = real_portion[:50, :]
    sns.heatmap(real_portion, linewidth=0.5)
    plt.show()

    portion = product_params_matrix.detach()
    portion = portion[:50, :]
    sns.heatmap(portion, linewidth=0.5)
    plt.show()

    return performance

def train_product_entire(df, max_scores, binarise_method, shuffle):
    
    cleaned_df = thres_score_range(df, max_scores)
    if binarise_method == 'mid':
        binarised_df = binarise_by_mid(cleaned_df, max_scores)
    elif binarise_method == 'avg':
        binarised_df = binarise_by_avg(cleaned_df)
    
    if shuffle:
        binarised_df = shuffle_cols(binarised_df)

    dataset_ts = torch.tensor(binarised_df.values)
    first_quadrant_ts, train_question_output_ts, train_student_output_ts, test_output_ts = split_to_4quadrants(dataset_ts, student_split=0.5, question_split=0.5)
    upper_half_ts = torch.cat([first_quadrant_ts, train_question_output_ts], dim=1)
    # test_nans = torch.tensor(float('nan')).repeat(test_output_ts.shape)
    # lower_half_ts = torch.cat([train_student_output_ts, test_nans], dim=1)
    # full_ts = torch.cat([upper_half_ts, lower_half_ts], dim=0)

    rows_split = int(dataset_ts.shape[0] * 0.5)
    cols_split = int(dataset_ts.shape[1] * 0.5)

    upper_half_mask = torch.ones(upper_half_ts.shape)
    # upper_left_zeros = torch.zeros(first_quadrant_ts.shape)
    # upper_right_ones = torch.ones(train_question_output_ts.shape)
    # upper_half_mask = torch.cat([upper_left_zeros, upper_right_ones], dim=1)

    bottom_left_ones = torch.ones(train_student_output_ts.shape)
    bottom_right_zeros = torch.zeros(test_output_ts.shape)
    bottom_half_mask = torch.cat([bottom_left_ones, bottom_right_zeros], dim=1)
    mask = torch.cat([upper_half_mask, bottom_half_mask], dim=0)

    learning_rate = 0.0001
    n_iters = 40000
    seed_number = 1000

    rng = torch.Generator()
    rng.manual_seed(seed_number)

    bs_tensor, bq_tensor, t_arr, nll_arr = train(learning_rate, n_iters, dataset_ts, mask, rng)
    plt.plot(t_arr, nll_arr)
    plt.title('Ability-difficulty-product baseline')
    plt.ylabel('Negative log likelihood')
    plt.xlabel('epoch')
    plt.show()

    # _, bq_tensor, t_arr, nll_arr = train(learning_rate, n_iters, train_question_output_ts)
    # plt.plot(t_arr, nll_arr)
    # plt.title('Training question params')
    # plt.ylabel('Negative log likelihood')
    # plt.xlabel('epoch')
    # plt.show()

    performance = predict(bs_tensor[rows_split:], bq_tensor[cols_split:], test_output_ts, rng)

    print(f"bs (student params): {bs_tensor}")
    print(f"bq (question params): {bq_tensor}")
    print(f"Percentage accuracy for product baseline: {performance}")
    # try optimise each dimension one by one, compute loss going along
    return performance