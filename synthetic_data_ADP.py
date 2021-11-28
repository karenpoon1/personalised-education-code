import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# from torch.autograd import Variable
from utils.split_data import split_to_4quadrants
from models.ability_difficulty_product import probit_correct, train_product_alternate_quadrants


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
    plt.xlabel('Questions')
    plt.ylabel('Students')
    plt.show()

    fake_data = torch.bernoulli(product_params_matrix, generator=rng)
    
    rng.manual_seed(1000)
    predictions_from_ground_truth = torch.bernoulli(product_params_matrix, generator=rng)

    _,_,_,fake_data_test_set = split_to_4quadrants(fake_data, student_split=0.5, question_split=0.5)
    _,_,_,predictions_from_ground_truth_test_set = split_to_4quadrants(predictions_from_ground_truth, student_split=0.5, question_split=0.5)

    performance = torch.sum(torch.eq(fake_data_test_set, predictions_from_ground_truth_test_set)) / torch.numel(fake_data_test_set)
    performance = float(performance)*100
    print(performance)

    fake_data_test_set_reshaped = fake_data_test_set.reshape(-1).type(torch.int)
    predictions_from_ground_truth_test_set_reshaped = predictions_from_ground_truth_test_set.reshape(-1).type(torch.int)

    conf_matrix = confusion_matrix(fake_data_test_set_reshaped.numpy(), predictions_from_ground_truth_test_set_reshaped.detach().numpy())
    conf_matrix = conf_matrix*100/torch.numel(fake_data_test_set)
    print(conf_matrix)
    
    real_portion = fake_data_test_set.detach()
    real_portion = real_portion[:50, :]
    sns.heatmap(real_portion, linewidth=0.5)
    plt.title('Synthesised data')
    plt.xlabel('Questions')
    plt.ylabel('Students')
    plt.show()

    predicted_portion = predictions_from_ground_truth_test_set.detach()
    predicted_portion = predicted_portion[:50, :]
    sns.heatmap(predicted_portion, linewidth=0.5)
    plt.title('Predicted output')
    plt.xlabel('Questions')
    plt.ylabel('Students')
    plt.show()

    return fake_data, bs_tensor_fake, bq_tensor_fake, portion


learning_rate = 0.0003
n_iters = 50000
seed_number = 1000

rng = torch.Generator()
rng.manual_seed(seed_number)

S = 45022
Q = 24

fake_data, bs_tensor_fake, bq_tensor_fake, portion_fake = synthesise_data_from_normal(S, Q, rng)
first_quadrant, train_question_output_ts, train_student_output_ts, test_output_ts = split_to_4quadrants(fake_data, student_split=0.5, question_split=0.5)

bs_tensor, bq_tensor, product_params_matrix, performance, conf_matrix = train_product_alternate_quadrants(train_question_output_ts, train_student_output_ts, test_output_ts, learning_rate, n_iters, rng)

portion = product_params_matrix.detach()
portion = portion[:50, :]
portion_diff = portion - portion_fake
sns.heatmap(portion_diff, linewidth=0.5)
plt.title('Error')
plt.xlabel('Questions')
plt.ylabel('Students')
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
