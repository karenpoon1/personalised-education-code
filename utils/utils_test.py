import unittest
import torch

from split_and_vectorise import split_and_vectorise

class TestUtils(unittest.TestCase):

    def test_split_and_vectorise(self):
        data_ts = torch.tensor([[1, 1, 0, 1, 0],
                                [1, 1, 0, 1, 0],
                                [1, 1, 0, 1, 0],
                                [0, 1, 1, 1, 0],
                                [0, 1, 0, 0, 0]],
                                dtype=torch.float)
        test_range = [[2,5], [2,5]]

        exp_train_vectorised_ts = torch.tensor([[1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4],
            [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 0, 1, 0, 1]], dtype=torch.int32)
        exp_test_vectorised_ts = torch.tensor([[0, 1, 0, 1, 1, 0, 0, 0, 0],
            [2, 2, 2, 3, 3, 3, 4, 4, 4],
            [2, 3, 4, 2, 3, 4, 2, 3, 4]], dtype=torch.int32)

        train_vectorised_ts, test_vectorised_ts = split_and_vectorise(data_ts, test_range, shuffle=False)
        self.assertTrue(torch.equal(train_vectorised_ts, exp_train_vectorised_ts) and torch.equal(test_vectorised_ts, exp_test_vectorised_ts))

if __name__ == '__main__':
    torch.manual_seed(0)
    unittest.main()
