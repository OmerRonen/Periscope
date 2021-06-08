import unittest

import numpy as np
import tensorflow as tf
from periscope.net.basic_ops import outer_concat, upper_triangular_cross_entropy_loss, layer_norm
from periscope.utils.utils import bin_array


class TestBasicOps(unittest.TestCase):
    def test_outer_concat(self):
        seq1 = np.random.random((1, 4, 7))
        seq1_tf = tf.constant(seq1)
        outer_concat_seq_tf = outer_concat(seq1_tf)
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
            assert np.isclose(val[0, 0, 0], np.sum(O_s[0, 0:2, ...] * weights0[1:3, :, 0]))

            assert np.isclose(val[0, 1, 0], np.sum(O_s[0, 0:3, ...] * weights0[..., 0]))

    def test_upper_triangular_cross_entropy_loss(self):
        distance_mat = np.random.normal(loc=10, scale=7, size=(20, 20))
        distance_mat[distance_mat < 0] = -1
        n_bins = 6
        alpha = 5
        binned_dm = bin_array(distance_mat, n_bins)

        binned_dm_tf = tf.constant(binned_dm)

        loss = upper_triangular_cross_entropy_loss(binned_dm_tf, binned_dm_tf, alpha)
        preds = 0.9 * binned_dm
        preds_tf = tf.constant(preds)
        loss2 = upper_triangular_cross_entropy_loss(preds_tf, binned_dm_tf, alpha)
        # expected loss
        valid = np.sum(distance_mat != -1)
        alpha_arr = np.ones_like(binned_dm)
        alpha_arr[..., 0:int(n_bins / 2)] = alpha
        expected_loss = -1 * alpha_arr * np.log(np.clip(preds, a_max=1, a_min=1e-10)) * binned_dm / valid
        expected_loss = np.sum(expected_loss, axis=2)
        diag_loss = np.sum(np.diag(expected_loss))
        expected_loss[np.tril_indices(20)] = 0
        expected_loss = np.sum(expected_loss) + diag_loss

        with tf.Session():
            assert loss.eval() == 0
            assert np.isclose(loss2.eval(), expected_loss)

    def test_layer_norm(self):
        conv_data = np.random.random((1, 3, 3, 5))
        conv_data_tf = tf.constant(conv_data)
        conv_data_tf_norm = layer_norm(conv_data_tf, unit_axis=3)
        with tf.Session():
            conv_data_norm = conv_data_tf_norm.eval()
        norm_data = (conv_data-np.mean(conv_data, axis=3, keepdims=True))/np.std(conv_data, axis=3, keepdims=True)
        assert np.allclose(conv_data_norm, norm_data)


if __name__ == '__main__':
    unittest.main()
