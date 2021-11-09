import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# from torch.autograd import Variable

seed_number = 1000
rng = torch.Generator()
rng.manual_seed(seed_number)

S = 45022
Q = 24
bs_tensor_fake = torch.normal(mean=0, std=np.sqrt(10), size=(S,), generator=rng)
bq_tensor_fake = torch.normal(mean=0, std=np.sqrt(10), size=(Q,), generator=rng)

def probit_correct(bs, bq):
    return 1/(1+torch.exp(-bs-bq))

# Generate fake data
bs_matrix = bs_tensor_fake.repeat(len(bq_tensor_fake), 1)
bs_matrix = torch.transpose(bs_matrix, 0, 1)
bq_matrix = bq_tensor_fake.repeat(len(bs_tensor_fake), 1)

product_params_matrix = probit_correct(bs_matrix, bq_matrix) # probit of correct answer

fake_data = torch.bernoulli(product_params_matrix, generator=rng)

# Split data
student_split = 0.5
question_split = 0.5
no_train_rows = int(S * student_split)
no_train_cols = int(Q * question_split)

upper_half, lower_half = torch.split(fake_data, no_train_rows, dim=0)
_, output_question_train = torch.split(upper_half, no_train_cols, dim=1)
output_student_train, output_test = torch.split(lower_half, no_train_cols, dim=1)

learning_rate = 0.0003
n_iters = 8000

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
    
    plt.plot(t_arr, nll_arr)
    plt.ylabel('Negative log likelihood')
    plt.xlabel('epoch')
    plt.show()

    return bs_tensor, bq_tensor

bs_tensor, _ = train(learning_rate, n_iters, output_student_train)
_, bq_tensor = train(learning_rate, n_iters, output_question_train)

# mse between params
# bs_mse = torch.sum((bs_tensor - bs_tensor_fake)**2)
# print(bs_mse)
# bq_mse =  torch.sum((bq_tensor - bq_tensor_fake)**2)
# print(bq_mse)

print('bs_tensor: ', bs_tensor)
print('bq_tensor: ', bq_tensor)

bs_matrix = bs_tensor.repeat(len(bq_tensor), 1)
bs_matrix = torch.transpose(bs_matrix, 0, 1)
bq_matrix = bq_tensor.repeat(len(bs_tensor), 1)

product_params_matrix = probit_correct(bs_matrix, bq_matrix)


predictions = torch.bernoulli(product_params_matrix, generator=rng)

performance = torch.sum(torch.eq(output_test, predictions)) / torch.numel(output_test)
performance = float(performance)*100

print(performance)
portion = product_params_matrix.detach()
portion = portion[:100, :]
ax = sns.heatmap(portion, linewidth=0.5)
plt.show()

# try optimise each dimension one by one, compute loss going along

