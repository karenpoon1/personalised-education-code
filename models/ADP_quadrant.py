import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def probit_correct(bs, bq):
    return 1/(1+torch.exp(-bs-bq))


def train(learning_rate, n_iters, output_tensor, rng):

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

        probit_1 = 1/(1+torch.exp(-bs_matrix-bq_matrix))
        probit_0 = 1 - probit_1
        nll = -torch.sum(output_tensor*torch.log(probit_1) + (1-output_tensor)*torch.log(probit_0))
        nll.backward()

        with torch.no_grad():
            bs_tensor -= learning_rate * bs_tensor.grad
            bq_tensor -= learning_rate * bq_tensor.grad

        # zero the gradients after updating
        bs_tensor.grad.zero_()
        bq_tensor.grad.zero_()

        if epoch % 100 == 0:
            print(epoch,nll)

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

    test_output_ts_reshaped = test_output_ts.reshape(-1).type(torch.int)
    predictions_reshaped = predictions.reshape(-1).type(torch.int)

    conf_matrix = confusion_matrix(test_output_ts_reshaped.numpy(), predictions_reshaped.detach().numpy())
    conf_matrix = conf_matrix*100/torch.numel(test_output_ts)
    
    real_portion = test_output_ts.detach()
    real_portion = real_portion[:50, :]
    sns.heatmap(real_portion, linewidth=0.5)
    plt.title('Real binarised data')
    plt.xlabel('Questions')
    plt.ylabel('Students')
    plt.show()

    predicted_probit_portion = product_params_matrix.detach()
    predicted_probit_portion = predicted_probit_portion[:50, :]
    sns.heatmap(predicted_probit_portion, linewidth=0.5)
    plt.title('Predicted probabilities')
    plt.xlabel('Questions')
    plt.ylabel('Students')
    plt.show()

    predicted_portion = predictions.detach()
    predicted_portion = predicted_portion[:50, :]
    sns.heatmap(predicted_portion, linewidth=0.5)
    plt.title('Predicted output')
    plt.xlabel('Questions')
    plt.ylabel('Students')
    plt.show()

    return product_params_matrix, performance, conf_matrix


def train_product_alternate_quadrants(first_train_quadrant_ts, second_train_quadrant_ts, test_output_ts, learning_rate, n_iters, rng):

    bs_tensor, _, t_arr, nll_arr = train(learning_rate, n_iters, second_train_quadrant_ts, rng)
    plt.plot(t_arr, nll_arr)
    plt.title('Training student params')
    plt.ylabel('Negative log likelihood')
    plt.xlabel('epoch')
    plt.show()

    _, bq_tensor, t_arr, nll_arr = train(learning_rate, n_iters, first_train_quadrant_ts, rng)
    plt.plot(t_arr, nll_arr)
    plt.title('Training question params')
    plt.ylabel('Negative log likelihood')
    plt.xlabel('epoch')
    plt.show()

    if len(bs_tensor) != test_output_ts.shape[0]:
        bs_tensor = bs_tensor[-test_output_ts.shape[0]:]
    if len(bq_tensor) != test_output_ts.shape[1]:
        bq_tensor = bq_tensor[-test_output_ts.shape[1]:]

    product_params_matrix, performance, conf_matrix = predict(bs_tensor, bq_tensor, test_output_ts, rng)
    return performance, conf_matrix


def train_product_upper_left(first_quadrant_ts, train_question_output_ts, train_student_output_ts, test_output_ts, learning_rate, n_iters, rng):
    upper_half_ts = torch.cat([first_quadrant_ts, train_question_output_ts], dim=1)
    left_half_ts = torch.cat([first_quadrant_ts, train_student_output_ts], dim=0)
    performance, conf_matrix = train_product_alternate_quadrants(upper_half_ts, left_half_ts, test_output_ts, learning_rate, n_iters, rng)
    return performance, conf_matrix
