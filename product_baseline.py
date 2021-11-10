import torch
import pandas as pd
import matplotlib.pyplot as plt

from utils.clean_data import thres_score_range
from utils.binarise_data import binarise_by_mid
from utils.shuffle_data import shuffle_cols
from utils.split_data import split_dataset

df = pd.read_csv("personalised-education/Fwd__Pinpoint_ML_Dataset/9to1_2017_GCSE_1H.csv", skiprows=list(range(1,24)), usecols=list(range(2,26)))
max_scores = pd.read_csv("personalised-education/Fwd__Pinpoint_ML_Dataset/9to1_2017_GCSE_1H.csv", nrows = 1, skiprows=[1], usecols=list(range(2,26)))

cleaned_df = thres_score_range(df, max_scores)
binarised_df = binarise_by_mid(cleaned_df, max_scores)
# binarised_df = shuffle_cols(binarised_df, shuffle_seed=0)

dataset_ts = torch.tensor(binarised_df.values)
first_quadrant, train_question_output_ts, train_student_output_ts, test_output_ts = split_dataset(dataset_ts, student_split=0.5, question_split=0.5)

learning_rate = 0.0003
n_iters = 60
seed_number = 1000

rng = torch.Generator()
rng.manual_seed(seed_number)

def train(learning_rate, n_iters, output_tensor):

    t_arr, nll_arr = [], []
    S = output_tensor.size()[0] # no. of rows
    Q = output_tensor.size()[1] # no. of cols

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
        nll = -torch.sum(output_tensor*probit_1 + (1-output_tensor)*probit_0)

        nll.backward()

        with torch.no_grad():
            bs_tensor -= learning_rate * bs_tensor.grad
            bq_tensor -= learning_rate * bq_tensor.grad

        # zero the gradients after updating
        bs_tensor.grad.zero_()
        bq_tensor.grad.zero_()

        if epoch % 100 == 0:
            print(nll)

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
    return performance

bs_tensor, _, t_arr, nll_arr = train(learning_rate, n_iters, train_student_output_ts)
plt.plot(t_arr, nll_arr)
plt.title('Training student params')
plt.ylabel('Negative log likelihood')
plt.xlabel('epoch')
# plt.show()

_, bq_tensor, t_arr, nll_arr = train(learning_rate, n_iters, train_question_output_ts)
plt.plot(t_arr, nll_arr)
plt.title('Training question params')
plt.ylabel('Negative log likelihood')
plt.xlabel('epoch')
# plt.show()

performance = predict(bs_tensor, bq_tensor, test_output_ts, rng)

print(f"bs (student params): {bs_tensor}")
print(f"bq (question params): {bq_tensor}")
print(f"Percentage accuracy for product baseline: {performance}")
# try optimise each dimension one by one, compute loss going along
