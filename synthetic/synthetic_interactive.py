import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import math

from utils.split_data import split_to_4quadrants
from models.I import I

def probit_correct(bs, bq, i):
    return 1/(1+torch.exp(-bs-bq-i))

def synthesise(S, Q, dimension, rng):
    # Generate random params
    bs_fake = torch.normal(mean=0, std=np.sqrt(10), size=(S,), generator=rng)
    bq_fake = torch.normal(mean=0, std=np.sqrt(10), size=(Q,), generator=rng)

    xs_fake = torch.randn((dimension,S), generator=rng)
    xq_fake = torch.randn((dimension,Q), generator=rng)

    # Reconstruct S x Q table
    xs_fake_0 = xs_fake[0].repeat(Q,1).T.reshape(-1)
    xs_fake_1 = xs_fake[1].repeat(Q,1).T.reshape(-1)

    xs_matrix = torch.stack((xs_fake_0,xs_fake_1))
    xq_matrix = xq_fake.repeat(1,S)

    interaction_vec = torch.sum((xs_matrix*xq_matrix), 0)
    interaction_matrix = interaction_vec.reshape(S,Q)

    bs_matrix = bs_fake.repeat(Q, 1)
    bs_matrix = torch.transpose(bs_matrix, 0, 1)
    bq_matrix = bq_fake.repeat(S, 1)

    product_params_matrix = probit_correct(bs_matrix, bq_matrix, interaction_matrix) # probit of correct answer

    student_split, question_split = 0.5, 0.5
    testset_start_row, testset_start_col = math.ceil(S*student_split), math.ceil(Q*question_split)

    # Visualise synthetic probabilities (ground truth)
    portion = product_params_matrix.detach()
    portion = portion[testset_start_row:testset_start_row+20, testset_start_col:]
    sns.heatmap(portion, linewidth=0.5)
    plt.title('Synthetic probabilities')
    plt.xlabel('Questions')
    plt.ylabel('Students')
    plt.show()

    # Generate synthetic data from ground truth
    fake_data = torch.bernoulli(product_params_matrix, generator=rng)
    
    # Generate predictions from ground truth
    ground_truth_predictions = torch.bernoulli(product_params_matrix, generator=rng)
    
    # Accuracy from ground truth
    _,_,_,fake_data_test_set = split_to_4quadrants(fake_data, student_split, question_split)
    _,_,_,ground_truth_predictions_test_set = split_to_4quadrants(ground_truth_predictions, student_split, question_split)

    performance = torch.sum(torch.eq(fake_data_test_set, ground_truth_predictions_test_set)) / torch.numel(fake_data_test_set)
    performance = float(performance)*100

    fake_data_test_set_reshaped = fake_data_test_set.reshape(-1).type(torch.int)
    ground_truth_predictions_test_set_reshaped = ground_truth_predictions_test_set.reshape(-1).type(torch.int)

    conf_matrix = confusion_matrix(fake_data_test_set_reshaped.numpy(), ground_truth_predictions_test_set_reshaped.detach().numpy())
    conf_matrix = conf_matrix*100/torch.numel(fake_data_test_set)
    
    real_portion = fake_data_test_set.detach()
    real_portion = real_portion[:30, :]
    sns.heatmap(real_portion, linewidth=0.5)
    plt.title('Synthesised data')
    plt.xlabel('Questions')
    plt.ylabel('Students')
    plt.show()

    predicted_portion = ground_truth_predictions_test_set.detach()
    predicted_portion = predicted_portion[:30, :]
    sns.heatmap(predicted_portion, linewidth=0.5)
    plt.title('Predicted output')
    plt.xlabel('Questions')
    plt.ylabel('Students')
    plt.show()

    return fake_data, bs_fake, bq_fake, performance, conf_matrix


# seed_number = 1001

# rng = torch.Generator()
# rng.manual_seed(seed_number)

# S = 45022
# Q = 24
# dimension = 2

# fake_data, bs_fake, bq_fake, performance, conf_matrix = synthesise(S, Q, dimension, rng)
# print(performance)
# print(conf_matrix)

# testset_row_range, testset_col_range = [int(S/2), S], [int(Q/2), Q]
# my_I = I(fake_data, testset_row_range, testset_col_range)
# my_I.run(learning_rate=0.00025, iters=1000, dimension=dimension, validation=0.9)

# # fake_data, bs_fake, bq_fake, portion_fake = synthesise_data_from_normal(100, 24, 2, rng)
