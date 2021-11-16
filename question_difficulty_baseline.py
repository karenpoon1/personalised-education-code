import torch

def train_question_difficulty(train_ts, test_ts, seed_number):
    rng = torch.Generator()
    rng.manual_seed(seed_number)

    no_train_rows = train_ts.shape[0]
    question_param_arr = torch.sum(train_ts, dim=0, keepdim=True) / no_train_rows
    x = question_param_arr.repeat(no_train_rows,1)
    predictions = torch.bernoulli(x, generator=rng)

    performance = torch.sum(torch.eq(test_ts, predictions)) / torch.numel(test_ts)
    performance = float(performance)*100
    return performance
