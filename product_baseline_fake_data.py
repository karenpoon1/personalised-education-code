import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# from torch.autograd import Variable
from utils.split_data import split_dataset

seed_number = 1000
rng = torch.Generator()
rng.manual_seed(seed_number)

S = 45022
Q = 24

learning_rate = 0.0003
n_iters = 80

def probit_correct(bs, bq):
    return 1/(1+torch.exp(-bs-bq))

def synthesise_data_from_normal(S, Q, rng):
    bs_tensor_fake = torch.normal(mean=0, std=np.sqrt(10), size=(S,), generator=rng)
    bq_tensor_fake = torch.normal(mean=0, std=np.sqrt(10), size=(Q,), generator=rng)

    bs_matrix = bs_tensor_fake.repeat(len(bq_tensor_fake), 1)
    bs_matrix = torch.transpose(bs_matrix, 0, 1)
    bq_matrix = bq_tensor_fake.repeat(len(bs_tensor_fake), 1)
    product_params_matrix = probit_correct(bs_matrix, bq_matrix) # probit of correct answer

    fake_data = torch.bernoulli(product_params_matrix, generator=rng)
    return fake_data

# Fit model
def train(learning_rate, n_iters, output_tensor):

    t_arr, nll_arr = [], []
    S = output_tensor.size()[0] # rows
    Q = output_tensor.size()[1] # cols

    bs_tensor = torch.randn(S, requires_grad=True, generator=rng)
    bq_tensor = torch.randn(Q, requires_grad=True, generator=rng)

    for epoch in range(n_iters):

        nll = 0
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

def predict(bs_tensor, bq_tensor, test_output_ts, rng):
    bs_matrix = bs_tensor.repeat(len(bq_tensor), 1)
    bs_matrix = torch.transpose(bs_matrix, 0, 1)
    bq_matrix = bq_tensor.repeat(len(bs_tensor), 1)

    product_params_matrix = probit_correct(bs_matrix, bq_matrix)

    predictions = torch.bernoulli(product_params_matrix, generator=rng)

    performance = torch.sum(torch.eq(test_output_ts, predictions)) / torch.numel(test_output_ts)
    performance = float(performance)*100

    portion = product_params_matrix.detach()
    portion = portion[:100, :]
    sns.heatmap(portion, linewidth=0.5)
    # plt.show()

    return performance

fake_data = synthesise_data_from_normal(S, Q, rng)
first_quadrant, train_question_output_ts, train_student_output_ts, test_output_ts = split_dataset(fake_data, student_split=0.5, question_split=0.5)

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

# mse between params
# bs_mse = torch.sum((bs_tensor - bs_tensor_fake)**2)
# bq_mse =  torch.sum((bq_tensor - bq_tensor_fake)**2)

# try optimise each dimension one by one, compute loss going along
