import unittest

import numpy as np
import tensorflow as tf
from periscope.net.basic_ops import _outer_concat


class TestOuterConcat(unittest.TestCase):
    def test_outer_concat(self):
        seq1 = np.random.random((1, 4, 7))
        seq1_tf = tf.constant(seq1)
        outer_concat_seq_tf = _outer_concat(seq1_tf)
        with tf.Session():
            outer_concat_seq = outer_concat_seq_tf.eval()
            assert outer_concat_seq[0, 2, 3, 4] == seq1[0, 2, 4]
            assert outer_concat_seq[0, 2, 3, 8] == seq1[0, 3, 1]


if __name__ == '__main__':
    unittest.main()
