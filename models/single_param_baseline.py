import torch

# df = pd.read_csv("Fwd__Pinpoint_ML_Dataset/9to1_2017_GCSE_1H.csv", skiprows=list(range(1,24)), usecols=list(range(2,26)))
# max_scores = pd.read_csv("Fwd__Pinpoint_ML_Dataset/9to1_2017_GCSE_1H.csv", nrows = 1, skiprows=[1], usecols=list(range(2,26)))

def train_single_param(first_quadrant_ts, train_question_ts, train_student_ts, test_ts, seed_number):

    rng = torch.Generator()
    rng.manual_seed(seed_number)

    no_ones_train = torch.sum(first_quadrant_ts) + torch.sum(train_question_ts) + torch.sum(train_student_ts)
    no_entries_train = torch.numel(first_quadrant_ts) + torch.numel(train_question_ts) + torch.numel(train_student_ts)
    single_param = no_ones_train / no_entries_train

    x = single_param.repeat(test_ts.shape[0], test_ts.shape[1])
    predictions = torch.bernoulli(x, generator=rng)

    performance = torch.sum(torch.eq(test_ts, predictions)) / torch.numel(test_ts)
    performance = float(performance)*100
    return performance
