import torch

def train_student_ability(train_ts, test_ts, seed_number):
    rng = torch.Generator()
    rng.manual_seed(seed_number)

    no_train_cols = train_ts.shape[1]
    student_param_tensor = torch.sum(train_ts, dim=1, keepdim=True) / no_train_cols
    x = student_param_tensor.repeat(1,no_train_cols)
    predictions = torch.bernoulli(x, generator=rng)

    performance = torch.sum(torch.eq(test_ts, predictions)) / torch.numel(test_ts)
    performance = float(performance)*100
    return performance
