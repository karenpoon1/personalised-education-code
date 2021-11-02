import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_student_df = pd.read_csv("split_dataset_cleaned/train_student_set.csv", usecols=list(range(1,13)))
train_question_df = pd.read_csv("split_dataset_cleaned/train_question_set.csv", usecols=list(range(1,13)))
test_df = pd.read_csv("split_dataset_cleaned/test_set.csv", usecols=list(range(1,13)))

n_training_data = 10000
train_student_df = train_student_df.head(n_training_data)
train_question_df = train_question_df.head(n_training_data)
test_df = test_df.head(n_training_data)

output_tensor_student = torch.tensor(train_student_df.values) # q1-q12, s22511-s23511
output_tensor_question = torch.tensor(train_question_df.values) # q13-q24, s1-s1000
output_tensor_test = torch.tensor(test_df.values) # q13-q24, s22511-s23511

learning_rate = 0.0005
n_iters = 5000

seed_number = 1000
rng = torch.Generator()
rng.manual_seed(seed_number)

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

        print(nll)

        t_arr.append(epoch)
        nll_arr.append(nll)
    
    plt.plot(t_arr, nll_arr)
    plt.ylabel('Negative log likelihood')
    plt.xlabel('epoch')
    plt.show()

    return bs_tensor, bq_tensor

bs_tensor, _ = train(learning_rate, n_iters, output_tensor_student)
_, bq_tensor = train(learning_rate, n_iters, output_tensor_question)

print(bs_tensor)
print(bq_tensor)

def probit_correct(bs, bq):
    return 1/(1+torch.exp(-bs-bq))

bs_matrix = bs_tensor.repeat(len(bq_tensor), 1)
bs_matrix = torch.transpose(bs_matrix, 0, 1)
bq_matrix = bq_tensor.repeat(len(bs_tensor), 1)

product_params_matrix = probit_correct(bs_matrix, bq_matrix)

predictions = torch.bernoulli(product_params_matrix, generator=rng)

performance = torch.sum(torch.eq(output_tensor_test, predictions)) / torch.numel(output_tensor_test)
performance = float(performance)*100

print(performance)
