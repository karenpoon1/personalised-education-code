import unittest
import torch

from split_and_vectorise import split_and_vectorise

class TestUtils(unittest.TestCase):

    def test_split_and_vectorise_1(self):
        data_ts = torch.tensor([[1, 2, 3, 4],
                                [5, 6, 7, 8],
                                [9, 10, 11, 12],
                                [13, 14, 15, 16]],
                                dtype=torch.float)
        testset_row_range, testset_col_range = [2,4], [2, 4] # Split matrix [row range, column range] out as test set

        # First row: data
        # Second row: row position
        # Third row: column position
        exp_train_vectorised_ts = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14],
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3],
            [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 0, 1]], dtype=torch.int32)
        exp_test_vectorised_ts = torch.tensor([[11, 12, 15, 16],
            [2, 2, 3, 3],
            [2, 3, 2, 3]], dtype=torch.int32)

        train_vectorised_ts, test_vectorised_ts = split_and_vectorise(data_ts, testset_row_range, testset_col_range, shuffle=False)
        self.assertTrue(torch.equal(train_vectorised_ts, exp_train_vectorised_ts) and torch.equal(test_vectorised_ts, exp_test_vectorised_ts))


    def test_split_and_vectorise_2(self):
        data_ts = torch.tensor([[1, 2, 3, 4],
                                [float('nan'), float('nan'), 7, 8],
                                [float('nan'), float('nan'), 11, 12],
                                [13, 14, 15, 16]],
                                dtype=torch.float)
        testset_row_range, testset_col_range = [0, 3], [1, 3] # Split matrix [row range, column range] out as test set

        # First row: data
        # Second row: row position
        # Third row: column position
        exp_train_vectorised_ts = torch.tensor([[1, 4, 8, 12, 13, 14, 15, 16],
            [0, 0, 1, 2, 3, 3, 3, 3],
            [0, 3, 3, 3, 0, 1, 2, 3]], dtype=torch.int32)
        exp_test_vectorised_ts = torch.tensor([[2, 3, 7, 11],
            [0, 0, 1, 2],
            [1, 2, 2, 2]], dtype=torch.int32)

        train_vectorised_ts, test_vectorised_ts = split_and_vectorise(data_ts, testset_row_range, testset_col_range, shuffle=False)
        self.assertTrue(torch.equal(train_vectorised_ts, exp_train_vectorised_ts) and torch.equal(test_vectorised_ts, exp_test_vectorised_ts))


if __name__ == '__main__':
    unittest.main()
