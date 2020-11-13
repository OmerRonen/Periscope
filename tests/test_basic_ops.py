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

    def test_conv_1d(self):
        O_s = np.random.random((1, 5, 4))  # tf.reshape(O, shape=(one, l, aa_dim * aa_dim))
        conv0_filt_shape = [
            3, 4, 1
        ]
        # initialise weights and bias for the filter
        weights0 = np.random.random(conv0_filt_shape)

        weights0_tf = tf.constant(weights0)
        O_s_tf = tf.constant(O_s)

        O_smooth = tf.nn.conv1d(input=O_s_tf, filters=weights0_tf, padding='SAME')
        with tf.Session():
            val = O_smooth.eval()
            assert np.isclose(val[0, 0, 0] ,np.sum(O_s[0,0:2, ...] * weights0[1:3,:,0]))

            assert np.isclose(val[0, 1, 0] ,np.sum(O_s[0,0:3, ...] * weights0[...,0]))


if __name__ == '__main__':
    unittest.main()
