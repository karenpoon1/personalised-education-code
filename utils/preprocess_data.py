import torch

from utils.clean_data import thres_score_range
from utils.binarise_data import binarise_by_mid, binarise_by_avg
from utils.shuffle_data import shuffle_cols

def process_raw(data_df, meta_df, binarise_method, shuffle):

    max_scores = meta_df.loc['Max'].astype(float)

    thres_df = thres_score_range(data_df, max_scores)
    if binarise_method == 'mid':
        binarised_df = binarise_by_mid(thres_df, max_scores)
    elif binarise_method == 'avg':
        binarised_df = binarise_by_avg(thres_df)
    
    if shuffle:
        binarised_df = shuffle_cols(binarised_df)

    return binarised_df.copy(), torch.clone(torch.tensor(binarised_df.values))
