import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("personalised-education/Fwd__Pinpoint_ML_Dataset/9to1_2017_GCSE_1H.csv", skiprows=list(range(1,24)), usecols=list(range(2,26)))
max_scores = pd.read_csv("personalised-education/Fwd__Pinpoint_ML_Dataset/9to1_2017_GCSE_1H.csv", nrows = 1, skiprows=[1], usecols=list(range(2,26)))

# clean data (scores above max go to max)
for col in df:
    max_score = max_scores[col].iloc[0]
    df.loc[df[col] > max_score, col] = max_score
    df.loc[df[col] < 0, col] = 0
    # binarise data by thresholding at mid score
    df[col] = (df[col] >= max_score/2).astype(float)

    # binarise data by thresholding at average
    # mean_score = df[col].mean()
    # df[col] = (df[col] >= mean_score).astype(float)

# shuffle data
shuffle_seed = 0
# df = df.sample(frac=1, axis=0, random_state=np.random.RandomState(shuffle_seed)) # shuffle rows
df = df.sample(frac=1, axis=1, random_state=np.random.RandomState(shuffle_seed)) # shuffle cols
df = df.reset_index(drop=True)

student_split = 0.5
question_split = 0.5
no_train_rows = int(len(df) * student_split)
no_train_cols = int(len(df.columns) * question_split)

train_question_df = df.iloc[:no_train_rows, no_train_cols:]
train_question_df = train_question_df.reset_index(drop=True)
train_student_df = df.iloc[no_train_rows:, :no_train_cols]
test_df = df.iloc[no_train_rows:, no_train_cols:]

# train_student_df = pd.read_csv("split_dataset_cleaned/train_student_set.csv", usecols=list(range(1,13)))
# train_question_df = pd.read_csv("split_dataset_cleaned/train_question_set.csv", usecols=list(range(1,13)))
# test_df = pd.read_csv("split_dataset_cleaned/test_set.csv", usecols=list(range(1,13)))

# n_training_data = 10000
# train_student_df = train_student_df.head(n_training_data)
# train_question_df = train_question_df.head(n_training_data)
# test_df = test_df.head(n_training_data)

output_tensor_student = torch.tensor(train_student_df.values) # q1-q12, s22511-s23511
output_tensor_question = torch.tensor(train_question_df.values) # q13-q24, s1-s1000
output_tensor_test = torch.tensor(test_df.values) # q13-q24, s22511-s23511

learning_rate = 0.0003
n_iters = 6000

seed_number = 1000
rng = torch.Generator()
rng.manual_seed(seed_number)

def train(learning_rate, n_iters, output_tensor, train_set):

    t_arr, nll_arr = [], []
    S = output_tensor.size()[0] # rows
    Q = output_tensor.size()[1] # cols

    bs_tensor = torch.randn(S, requires_grad=True, generator=rng)
    bq_tensor = torch.randn(Q, requires_grad=True, generator=rng)
    
    # bs_tensor = Variable(torch.normal(mean=0, std=np.sqrt(10), size=(S,), generator=rng), requires_grad=True)
    # bq_tensor = Variable(torch.normal(mean=0, std=np.sqrt(10), size=(Q,), generator=rng), requires_grad=True)

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
    if train_set == 's':
        plt.title('Training student params')
    elif train_set == 'q':
        plt.title('Training question params')
    plt.ylabel('Negative log likelihood')
    plt.xlabel('epoch')
    plt.show()

    return bs_tensor, bq_tensor

bs_tensor, _ = train(learning_rate, n_iters, output_tensor_student, 's')
_, bq_tensor = train(learning_rate, n_iters, output_tensor_question, 'q')

print(bs_tensor)
print(bq_tensor)

bs_tensor_df = pd.DataFrame(bs_tensor)
bq_tensor_df = pd.DataFrame(bq_tensor)
bs_tensor_df.to_csv('trained_bs_tensor.csv')
bq_tensor_df.to_csv('trained_bq_tensor.csv')

def probit_correct(bs, bq):
    return 1/(1+torch.exp(-bs-bq))

bs_matrix = bs_tensor.repeat(len(bq_tensor), 1)
bs_matrix = torch.transpose(bs_matrix, 0, 1)
bq_matrix = bq_tensor.repeat(len(bs_tensor), 1)

product_params_matrix = probit_correct(bs_matrix, bq_matrix)
product_params_matrix_df = pd.DataFrame(product_params_matrix)
product_params_matrix_df.to_csv('product_params_matrix.csv')

predictions = torch.bernoulli(product_params_matrix, generator=rng)

performance = torch.sum(torch.eq(output_tensor_test, predictions)) / torch.numel(output_tensor_test)
performance = float(performance)*100

print(performance)

# try optimise each dimension one by one, compute loss going along
# generate fake data