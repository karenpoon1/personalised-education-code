import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.clean_data import thres_score_range
from utils.binarise_data import binarise_by_mid, binarise_by_avg
from utils.shuffle_data import shuffle_cols
from utils.split_data import split_to_4quadrants

df = pd.read_csv("personalised-education-public/Fwd__Pinpoint_ML_Dataset/9to1_2017_GCSE_1H.csv", skiprows=list(range(1,24)), usecols=list(range(2,26)))
max_scores = pd.read_csv("personalised-education-public/Fwd__Pinpoint_ML_Dataset/9to1_2017_GCSE_1H.csv", nrows = 1, skiprows=[1], usecols=list(range(2,26)))
meta_diff = pd.read_csv("personalised-education-public/Fwd__Pinpoint_ML_Dataset/9to1_2017_GCSE_1H.csv", nrows = 1, skiprows=list(range(1,4)), usecols=list(range(2,26)))

# Test portions of data
df = df.head(5000)

def train(learning_rate, n_iters, output_tensor, dmeta, rng):

    t_arr, nll_arr = [], []
    S = output_tensor.size()[0] # no. of rows
    Q = output_tensor.size()[1] # no. of cols

    bs_tensor = torch.randn(S, requires_grad=True, generator=rng)
    bq_tensor = torch.randn(Q, requires_grad=True, generator=rng)
    w_q = torch.randn(Q, requires_grad=True, generator=rng)
    
    # bs_tensor = Variable(torch.normal(mean=0, std=np.sqrt(10), size=(S,), generator=rng), requires_grad=True)
    # bq_tensor = Variable(torch.normal(mean=0, std=np.sqrt(10), size=(Q,), generator=rng), requires_grad=True)

    for epoch in range(n_iters):

        bs_matrix = bs_tensor.repeat(Q, 1)
        bs_matrix = torch.transpose(bs_matrix, 0, 1)
        bq_matrix = bq_tensor.repeat(S, 1)

        dmeta_bias = dmeta*w_q
        dmeta_matrix = dmeta_bias.repeat(S, 1)

        probit_1 = torch.log(1/(1+torch.exp(-bs_matrix-bq_matrix-dmeta_matrix)))
        probit_0 = torch.log(1/(1+torch.exp(+bs_matrix+bq_matrix+dmeta_matrix)))
        nll = -torch.sum(output_tensor*probit_1 + (1-output_tensor)*probit_0)

        nll.backward()

        with torch.no_grad():
            bs_tensor -= learning_rate * bs_tensor.grad
            bq_tensor -= learning_rate * bq_tensor.grad
            w_q -= learning_rate * w_q.grad

        # zero the gradients after updating
        bs_tensor.grad.zero_()
        bq_tensor.grad.zero_()
        w_q.grad.zero_()

        if epoch % 100 == 0:
            print(epoch,nll)

        t_arr.append(epoch)
        nll_arr.append(nll)

    return bs_tensor, bq_tensor, w_q, t_arr, nll_arr

def probit_correct(bs, bq, d_bias):
    return 1/(1+torch.exp(-bs-bq-d_bias))

def predict(bs_tensor, bq_tensor, test_output_ts, rng, w_q, dmeta):
    bs_matrix = bs_tensor.repeat(len(bq_tensor), 1)
    bs_matrix = torch.transpose(bs_matrix, 0, 1)
    bq_matrix = bq_tensor.repeat(len(bs_tensor), 1)

    d_bias = dmeta*w_q
    d_bias_matrix = d_bias.repeat(len(bs_tensor), 1)

    product_params_matrix = probit_correct(bs_matrix, bq_matrix, d_bias_matrix)

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

    return product_params_matrix, performance

def train_meta(df, max_scores, meta_diff, binarise_method, shuffle):
    
    cleaned_df = thres_score_range(df, max_scores)
    if binarise_method == 'mid':
        binarised_df = binarise_by_mid(cleaned_df, max_scores)
    elif binarise_method == 'avg':
        binarised_df = binarise_by_avg(cleaned_df)
    
    if shuffle:
        binarised_df = shuffle_cols(binarised_df)

    dataset_ts = torch.tensor(binarised_df.values)
    _, train_question_output_ts, train_student_output_ts, test_output_ts = split_to_4quadrants(dataset_ts, student_split=0.5, question_split=0.5)
 
    # meta_diff_ts = torch.tensor(meta_diff.values)
    meta_diff_ts = torch.tensor([meta_diff.values])
    first_half_dmeta, second_half_dmeta = torch.split(meta_diff_ts, int(dataset_ts.shape[1]*0.5), dim=1)

    learning_rate = 0.00005
    n_iters = 60000
    seed_number = 1000

    rng = torch.Generator()
    rng.manual_seed(seed_number)

    bs_tensor, _, _, t_arr, nll_arr = train(learning_rate, n_iters, train_student_output_ts, first_half_dmeta, rng)
    _, bq_tensor, w_q, t_arr2, nll_arr2 = train(learning_rate, n_iters, train_question_output_ts, second_half_dmeta, rng)

    plt.plot(t_arr, nll_arr)
    plt.title('Training student params')
    plt.ylabel('Negative log likelihood')
    plt.xlabel('epoch')
    plt.show()

    plt.plot(t_arr2, nll_arr2)
    plt.title('Training question params')
    plt.ylabel('Negative log likelihood')
    plt.xlabel('epoch')
    plt.show()

    product_params_matrix, performance = predict(bs_tensor, bq_tensor, test_output_ts, rng, w_q, second_half_dmeta)

    print(f"bs (student params): {bs_tensor}")
    print(f"bq (question params): {bq_tensor}")
    print(f"w_q (diff param): {w_q}")
    print(f"Predicted probabilities: {product_params_matrix}")
    print(f"Percentage accuracy for product baseline: {performance}")
    # try optimise each dimension one by one, compute loss going along
    return