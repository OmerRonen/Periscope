import logging
import os
import tensorflow as tf
import numpy as np

from periscope.net.basic_ops import (upper_triangular_cross_entropy_loss, upper_triangular_mse_loss,
                                     get_top_category_accuracy, compare_predictions, _attention_op)
from periscope.net.contact_map import ContactMapEstimator
from periscope.net.params import NetParams
from periscope.utils.constants import FEATURES, ARCHS

LOCAL = os.getcwd() == '/Users/omerronen/Documents/MSA-Completion'

LOGGER = logging.getLogger(__name__)


def test_loss_functions():
    predicted_contact = np.random.random((2, 2, 2))
    predicted_contact = predicted_contact / predicted_contact.sum()
    predicted_contact_tf = tf.constant(predicted_contact)
    contact = np.ones((2, 2, 1))
    contact[0, 1, 0] = -1
    contact_tf = tf.constant(contact, dtype=tf.float64)

    loss_ce = upper_triangular_cross_entropy_loss(predicted_contact_tf,
                                                  contact_tf,
                                                  alpha=1)
    loss_mse = upper_triangular_mse_loss(predicted_contact_tf, contact_tf)
    loss_mse_zero = upper_triangular_mse_loss(contact_tf, contact_tf)
    loss_ce_zero = upper_triangular_cross_entropy_loss(contact_tf,
                                                       contact_tf,
                                                       alpha=2)
    with tf.Session():
        expected_loss_arr = -1 * (np.log(predicted_contact[:, :, 0]) *
                                  (contact[:, :, 0]) +
                                  np.log(1 - predicted_contact[:, :, 0]) *
                                  (1 - contact[:, :, 0]))
        mse_loss_arr = (predicted_contact[:, :, 0] - contact[:, :, 0])**2
        actual_loss_ce = loss_ce.eval()
        expected_loss_ce = (expected_loss_arr[1, 1] +
                            expected_loss_arr[0, 0]) / 3
        assert np.isclose(actual_loss_ce, expected_loss_ce)
        assert np.isclose(
            loss_mse.eval(),
            (mse_loss_arr[0, 1] + mse_loss_arr[1, 1] + mse_loss_arr[0, 0]) / 4)
        assert np.isclose(loss_mse_zero.eval(), 0)

        assert loss_ce_zero.eval() < loss_ce.eval()

        assert np.isclose(loss_ce_zero.eval(), 0)


def test_get_top_category_accuracy():
    predictions = tf.constant(np.random.random((1, 7, 7)), dtype=tf.float32)
    contact_map_np = np.random.randint(low=0, high=2, size=(1, 7, 7, 1))
    contact_map_np[0, 0, 6, 0] = 1
    contact_map = tf.constant(contact_map_np, dtype=tf.float32)
    sequence_length = tf.constant(7, dtype=tf.int32)
    acc = get_top_category_accuracy(category='S',
                                    top=7,
                                    predictions=predictions,
                                    contact_map=contact_map,
                                    sequence_length=sequence_length,
                                    mode='train')
    with tf.Session():
        acc_tf = acc.eval()
        assert np.isclose(acc_tf, contact_map_np[0, 0, 6, 0])


def test_compare_predictions():
    pred_a_np = np.random.randint(low=0, high=2, size=(1, 7, 7, 1))
    pred_b_np = pred_a_np
    pred_b_np[0, 0, 6, 0] = 1
    pred_a_np[0, 0, 6, 0] = 1
    sequence_length = tf.constant(7, dtype=tf.int32)

    acc = compare_predictions(tf.constant(pred_a_np), tf.constant(pred_b_np),
                              sequence_length, 'train')
    with tf.Session():
        acc_tf = acc.eval()
        assert acc_tf == 1


def test_alignemnt_attention_op():
    l = 100
    n = 500
    input_tensor = tf.constant(np.random.random((1, n, l)), dtype=tf.float32)
    q_dim = 5
    v_dim = 10
    attn_op = _attention_op(input_tensor=input_tensor,
                            q_dim=q_dim,
                            v_dim=v_dim,
                            name='attention',
                            num_homologous=n)

    assert attn_op.shape == (1, v_dim, l)


def test_train_cnn_random(tmpdir):
    params = NetParams.generate_net_params(name=tmpdir,
                                           conv_features=[FEATURES.ccmpred, FEATURES.evfold],
                                           resnet_features=[],
                                           arch=ARCHS.conv,
                                           k=10,
                                           save_summary_steps=2,
                                           save_checkpoints_steps=20,
                                           batch_size=1,
                                           epochs=3,
                                           num_bins=2,
                                           num_channels=10,
                                           filter_shape=(30, 30),
                                           dilation=1,
                                           num_layers=4,
                                           train_dataset=None,
                                           eval_dataset=None)

    if LOCAL:
        cnn_pred = ContactMapEstimator(params)
        cnn_pred.train_and_evaluate()


def test_train_cnn_live_0(tmpdir):
    params = NetParams.generate_net_params(
        name=tmpdir,
        conv_features=[FEATURES.reference_dm, FEATURES.ccmpred, FEATURES.evfold],
        resnet_features=[],
        arch=ARCHS.conv,
        save_summary_steps=1,
        save_checkpoints_steps=1,
        epochs=1,
        num_bins=2,
        batch_size=1,
        num_channels=10,
        filter_shape=(5, 5),
        dilation=1,
        num_layers=2,
        k=9,
        lr=0.001,
        train_dataset='testing',
        eval_dataset='testing',
        test_dataset='testing')

    if not LOCAL:
        cnn_pred = ContactMapEstimator(params)
        cnn_pred.train_and_evaluate()

def test_train_cnn_live_1(tmpdir):
    params = NetParams.generate_net_params(
        name=tmpdir,
        conv_features=[FEATURES.ccmpred, FEATURES.reference_dm, FEATURES.evfold],
        resnet_features=[],
        arch=ARCHS.multi_structure_ccmpred,
        save_summary_steps=1,
        save_checkpoints_steps=1,
        epochs=1,
        num_bins=2,
        batch_size=1,
        num_channels=10,
        filter_shape=(5, 5),
        dilation=1,
        num_layers=2,
        k=9,
        lr=0.001,
        train_dataset='testing',
        eval_dataset='testing',
        test_dataset='testing')

    if not LOCAL:
        cnn_pred = ContactMapEstimator(params)
        cnn_pred.train_and_evaluate()

