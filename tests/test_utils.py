import unittest

import numpy as np

from periscope.utils.utils import bin_array


class TestUtils(unittest.TestCase):
    def test_bin_array(self):
        distance_matrix = np.array([[0, 25], [7, 0]])
        binned2 = bin_array(distance_matrix, 2)
        assert np.allclose(binned2, np.array([[1, 0], [1, 1]]))

        distance_matrix = np.array([[-1, 25], [7, 0]])

        binned4 = bin_array(distance_matrix, 4)
        # we add na bin
        assert binned4.shape[2] == 4
        # one na
        first_bin = np.array([[1, 0], [0, 0]])
        # 0-4
        second_bin = np.array([[0, 0], [0, 1]])
        #  4-8
        third_bin = np.array([[0, 0], [1, 0]])
        #  8-24
        forth_bin = np.array([[0, 0], [0, 0]])
        # 24-inf
        last_bin = np.array([[0, 1], [0, 0]])

        binned_arr_expected = np.stack([second_bin, third_bin, forth_bin, last_bin], axis=2)

        assert np.allclose(binned4, binned_arr_expected)


if __name__ == '__main__':
    unittest.main()
