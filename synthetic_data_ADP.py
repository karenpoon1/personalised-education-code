import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# from torch.autograd import Variable
from utils.split_data import split_to_4quadrants
from ability_difficulty_product import probit_correct, train_product_alternate_quadrants


def synthesise_data_from_normal(S, Q, rng):
    bs_tensor_fake = torch.normal(mean=0, std=np.sqrt(10), size=(S,), generator=rng)
    bq_tensor_fake = torch.normal(mean=0, std=np.sqrt(10), size=(Q,), generator=rng)

    bs_matrix = bs_tensor_fake.repeat(len(bq_tensor_fake), 1)
    bs_matrix = torch.transpose(bs_matrix, 0, 1)
    bq_matrix = bq_tensor_fake.repeat(len(bs_tensor_fake), 1)
    product_params_matrix = probit_correct(bs_matrix, bq_matrix) # probit of correct answer

    portion = product_params_matrix.detach()
    portion = portion[22511:22511+50, 12:]
    sns.heatmap(portion, linewidth=0.5)
    plt.title('Synthetic probabilities')
    plt.show()

    fake_data = torch.bernoulli(product_params_matrix, generator=rng)
    return fake_data, bs_tensor_fake, bq_tensor_fake, portion


learning_rate = 0.0003
n_iters = 80
seed_number = 1000

rng = torch.Generator()
rng.manual_seed(seed_number)

S = 45022
Q = 24

fake_data, bs_tensor_fake, bq_tensor_fake, portion_fake = synthesise_data_from_normal(S, Q, rng)
first_quadrant, train_question_output_ts, train_student_output_ts, test_output_ts = split_to_4quadrants(fake_data, student_split=0.5, question_split=0.5)

bs_tensor, bq_tensor, product_params_matrix, performance = train_product_alternate_quadrants(train_question_output_ts, train_student_output_ts, test_output_ts, learning_rate, n_iters, rng)

portion = product_params_matrix.detach()
portion = portion[:50, :]
portion_diff = portion - portion_fake
sns.heatmap(portion_diff, linewidth=0.5)
plt.title('Error')
plt.show()

# print(f"bs_fake (student params): {bs_tensor_fake[22511:22511+50]}")
# print(f"bq_fake (question params): {bq_tensor_fake[12:]}")

# print((bs_tensor - bs_tensor_fake[int(S*0.5):])**2)
# print((bq_tensor - bq_tensor_fake[int(Q*0.5):])**2)

# # mse between params
# bs_mse = torch.sum((bs_tensor - bs_tensor_fake[int(S*0.5):])**2)
# bq_mse =  torch.sum((bq_tensor - bq_tensor_fake[int(Q*0.5):])**2)
# print(bs_mse)
# print(bq_mse)
