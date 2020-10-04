import os
import logging
import warnings

import tensorflow as tf

from ..utils.constants import (C_MAP_PRED, UPPER_TRIANGULAR_MSE_LOSS,
                               UPPER_TRIANGULAR_CE_LOSS,
                               PROTEIN_BOW_DIM, N_LAYERS_PROJ)
from ..utils.utils import compute_sequence_distance_mat_np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

BIN_THRESHOLDS = {1: 8.5, 2: 1}

LOGGER = logging.getLogger(__name__)
warnings.filterwarnings('ignore',  module='/tensorflow/')


def flatten_tf(tensor):
    return tf.reshape(tensor, [-1])


def _get_num_refrences_np(l):
    l_int = int(l)
    return l_int


@tf.function(input_signature=[tf.TensorSpec((), dtype=tf.int32)])
def _get_num_refrences(l):
    num_refrences = tf.numpy_function(_get_num_refrences_np, [l], tf.float32)
    return num_refrences


@tf.function(input_signature=[tf.TensorSpec((), dtype=tf.int32)])
def _compute_sequence_distance_mat(l):
    sequence_distance_ind_mat = tf.numpy_function(
        compute_sequence_distance_mat_np, [l], tf.float32)
    return sequence_distance_ind_mat


def compare_predictions(pred_a, pred_b, sequence_length, mode):
    """Compares the similarity of prediciton b to preidiction

    We calculate how many of method a contacts were also predicted by methods b, we assume the higher score means
    more likely to be in contact for both methods

    Args:
        pred_a (tf.Tensor): prediction matrix of method a
        pred_b (tf.Tensor): prediction matrix of method b
        sequence_length (tf.Tensor): length of the sequence shape 1
        mode (str): 'train', 'eval', or 'predict'


    Returns:
        tf.Tensor: prediction score

    """

    pred_a_flat = flatten_tf(_mask_lower_triangle(pred_a))
    pred_b_flat = flatten_tf(_mask_lower_triangle(pred_b))

    indices_distance_matrix_flat = flatten_tf(
        _compute_sequence_distance_mat(sequence_length))
    valid_inds = indices_distance_matrix_flat >= 6
    n_preds = tf.cast(
        tf.math.ceil(tf.reduce_sum(tf.cast(valid_inds, tf.float32)) * 0.05),
        tf.int32)

    pred_a_cat = pred_a_flat[valid_inds]
    pred_b_cat = pred_b_flat[valid_inds]

    combined_arr = tf.stack([pred_a_cat, pred_b_cat], axis=1)
    reordered = tf.gather(combined_arr,
                          tf.nn.top_k(combined_arr[:, 0], k=n_preds).indices)
    reordered_top = reordered[:, 1]
    b_threshold = tf.math.top_k(
        pred_b_cat, k=n_preds,
        sorted=True)[0][-1] * tf.ones_like(reordered_top)

    preds = tf.cast(reordered_top >= b_threshold, tf.float32)
    if mode == tf.estimator.ModeKeys.TRAIN:

        return tf.reduce_mean(preds)
    elif mode == tf.estimator.ModeKeys.EVAL:
        return tf.metrics.mean(preds)


def get_top_category_accuracy(category, top, predictions, contact_map,
                              sequence_length, mode):
    """Computes the average accuracy of top prediction in a distance category

    Args:
        category (str): the distance category can be "S", "M" or "L"
        top (int): the number to divide l by
        predictions (tf.Tensor): contact prediction matrix of shape 1 * l * l * 1
        contact_map (tf.Tensor): contact label matrix of shape 1 * l * l * 1
        sequence_length (tf.Tensor): length of the sequence shape 1
        mode (str): 'train', 'eval', or 'predict'

    Returns:
        tf.Tensor: accuracy of shape 1
    """

    contact_map_dims = tf.expand_dims(tf.expand_dims(tf.squeeze(contact_map), -1), 0)
    predictions_dims = tf.expand_dims(tf.expand_dims(tf.squeeze(predictions), -1), 0)

    c_map = tf.where(tf.less(contact_map_dims, 0), tf.zeros_like(contact_map_dims), contact_map_dims)
    preds = tf.where(tf.less(contact_map_dims, 0), tf.zeros_like(contact_map_dims), predictions_dims)

    predictions_flat = flatten_tf(_mask_lower_triangle(preds))
    contact_map_flat = flatten_tf(_mask_lower_triangle(c_map))

    indices_distance_matrix_flat = flatten_tf(
        _compute_sequence_distance_mat(sequence_length))
    top_k = sequence_length // top

    def _get_category_ind():
        if category == 'S':
            return (indices_distance_matrix_flat >=
                    6) & (indices_distance_matrix_flat < 12)
        elif category == 'M':
            return (indices_distance_matrix_flat >=
                    12) & (indices_distance_matrix_flat < 24)
        elif category == 'L':
            return indices_distance_matrix_flat >= 24

    def _get_true_rate(true, pred, n_inds, mode):
        combined_arr = tf.stack([pred, true], axis=1)

        k = tf.reduce_min([n_inds, top_k])

        reordered = tf.gather(combined_arr,
                              tf.nn.top_k(combined_arr[:, 0], k=k).indices)
        preds = reordered[:, 1]
        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.PREDICT:
            return tf.reduce_mean(preds)
        elif mode == tf.estimator.ModeKeys.EVAL:
            return tf.metrics.mean(preds)

    category_ind = _get_category_ind()

    n_inds = tf.reduce_sum(tf.cast(category_ind, tf.int32))

    category_true = contact_map_flat[category_ind]
    category_pred = predictions_flat[category_ind]

    return _get_true_rate(category_true, category_pred, n_inds, mode)


def _attention_params():
    params = {"num_layers": 5, "q_dim": 20, "v_dim": 5}
    return params


def pairwise_conv_layer(input_data,
                        num_features,
                        num_filters,
                        filter_shape,
                        dilation,
                        name,
                        residual=False):
    """Applies 2-d convolution with biasing and relu activation

    Args:
        input_data (tf.Tensor): input to conv layer of shape: batch * l * l * num_features
        num_features (int): number of pairwise _features
        num_filters (int): number of output channels
        filter_shape (tuple[int, int]): shape of convolution filter
        dilation (int):  dilation factor
        name (str): layer name
        residual (bool): if true we use residual arch by summing the input with the output

    Returns:
        tf.Tensor: tensor of shape batch * l * l * num_filters

    """

    # setup the filter input shape for tf.nn.conv_2d

    if residual:
        num_filters = num_features

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        conv_filt_shape = [
            filter_shape[0], filter_shape[1], num_features, num_filters
        ]
        # initialise weights and bias for the filter

        weights = tf.get_variable(
            "W",
            shape=conv_filt_shape,
            initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable(
            "b",
            shape=[num_filters],
            initializer=tf.contrib.layers.xavier_initializer())

        # weights = tf.Variable(tf.truncated_normal(conv_filt_shape,mean=1,
        #                                           stddev=0.3),
        #                       name='W',
        #                       dtype=tf.float32,use_resource=True).initialized_value()
        # bias = tf.Variable(tf.truncated_normal([num_filters], mean=1, stddev=0.3),
        #                    name='b',use_resource=True).initialized_value()

        # setup the convolutional layer operation

        activated_input = tf.nn.tanh(input_data) if residual else input_data

        out_layer = tf.nn.conv2d(activated_input,
                                 weights, [1, 1, 1, 1],
                                 dilations=(1, dilation, dilation, 1),
                                 padding='SAME')

        # add the bias
        out_layer += bias

        # apply a ReLU non-linear activation
        out_layer = tf.nn.relu(out_layer, name='out')

        if residual:
            out_layer += input_data

        return out_layer


def upper_triangular_cross_entropy_loss(predicted_contact_map,
                                        contact_map,
                                        alpha=5):
    """Computes binary cross entropy loss over upper triangle of a tensor

    Args:
        predicted_contact_map (tf.Tensor): probability predictions shape 1 * l * l * 2
        contact_map (tf.Tensor): contact shape 1 * l * l * 1
        alpha (float): multiplication factor for the contact class

    Returns:
        tf.Tensor: loss value

    """

    predicted_contact_map_2d = tf.clip_by_value(tf.squeeze(
        predicted_contact_map[..., 0]),
        clip_value_min=1e-10,
        clip_value_max=1)
    inverse_predicted_contact_map_2d = tf.clip_by_value(
        1 - tf.squeeze(predicted_contact_map[..., 0]),
        clip_value_min=1e-10,
        clip_value_max=1)

    contact_map_2d = tf.squeeze(contact_map[..., 0])
    cm_ones = tf.ones_like(contact_map_2d)
    cm_zeros = tf.zeros_like(contact_map_2d)
    valid_inds = tf.where(tf.less(contact_map_2d, 0), cm_zeros, cm_ones)

    loss_matrix = -1 * (
            alpha * tf.log(predicted_contact_map_2d) * contact_map_2d +
            tf.log(inverse_predicted_contact_map_2d) * (1 - contact_map_2d))

    loss_matrix = tf.where(tf.equal(valid_inds, 0), cm_zeros, loss_matrix)

    loss_matrix_masked = flatten_tf(_mask_lower_triangle(loss_matrix))

    upper_triangular_loss = tf.reduce_sum(
        loss_matrix_masked, name=UPPER_TRIANGULAR_CE_LOSS) / tf.reduce_sum(
        flatten_tf(valid_inds))

    return upper_triangular_loss


def _mask_lower_triangle(tensor):
    """masks the lower traingle value of a tensor

    Args:
        tensor (tf.Tensor): the tensor matrix to mask

    Returns:
        tf.Tensor: same shape as input, masked

    """

    matrix_tf = tf.squeeze(tensor)

    ones = tf.ones_like(matrix_tf)
    mask_a = tf.matrix_band_part(ones, 0,
                                 -1)  # Upper triangular matrix of 0s and 1s
    mask = tf.cast(mask_a, dtype=tensor.dtype)  # Make a bool mask
    return matrix_tf * mask


def upper_triangular_mse_loss(predicted_contact_map, contact_map):
    """Computes binary cross entropy loss over upper triangle of a tensor

    Args:
        predicted_contact_map (tf.Tensor): probability predictions shape 1 * l * l * 1
        contact_map (tf.Tensor): contact shape 1 * l * l * 1

    Returns:
        tf.Tensor: loss value

    """

    contact_map_2d = contact_map[..., 0]
    predicted_contact_map_2d = predicted_contact_map[..., 0]

    loss_matrix = (predicted_contact_map_2d - contact_map_2d) ** 2

    loss_matrix = tf.where(contact_map_2d == -1, tf.zeros_like(loss_matrix),
                           loss_matrix)

    loss_matrix_masked = flatten_tf(_mask_lower_triangle(loss_matrix))

    upper_triangular_loss = tf.reduce_mean(loss_matrix_masked,
                                           name=UPPER_TRIANGULAR_MSE_LOSS)

    return upper_triangular_loss


def batch_loss(predicted_contact_maps, contact_maps, loss_fn):
    """Computes the average loss of a batch of _proteins

    Args:
        predicted_contact_maps (list[tf.Tensor]): list of predicted contact maps
        contact_maps (list[tf.Tensor]): list of true contact maps
        loss_fn (function):

    Returns:
        tf.Tensor: loss value for the batch

    """
    losses = []

    for i in range(len(predicted_contact_maps)):
        cm_pred = predicted_contact_maps[i]
        cm = contact_maps[i]
        losses.append(loss_fn(cm_pred, cm))

    average_loss = tf.reduce_mean(losses)
    return average_loss


def multi_structures_op(dms, seq_refs, seq_target, evfold, conv_params, ccmpred=None, k=30, deep_projection=False):
    """Multi structures contact map prediction operation

    Args:
        dms (tf.Tensor): reference distance matrix of shape (1, l, l, None)
        seq_refs (tf.Tensor): reference sequence of shape (1, l, PROTEIN_BOW_DIM, None)
        seq_target (tf.Tensor): target sequence of shape (1, l , PROTEIN_BOW_DIM)
        evfold (tf.Tensor): evfold energy matrix of shape (1, l, l, 1)
        conv_params (dict): parameters to the last deep conv layer
        ccmpred (tf.Tensor): ccmpred energy matrix of shape (1, l, l, 1)
        k (int): dimension to project the sequence to
        prot_dim (int): dimension of the protein numeric representation

    Returns:
        tf.Tensor: predicted contact map of shape (1, l,  l, num_bins)

    """
    structures_info = []
    prot_dim = PROTEIN_BOW_DIM
    for i in range(dms.get_shape().as_list()[3]):
        structures_info.append(
            _single_structure_op(dm=dms[..., i:i + 1],
                                 seq_ref=seq_refs[..., i],
                                 seq_target=seq_target,
                                 evfold=evfold,
                                 ccmpred=ccmpred,
                                 k=k,
                                 prot_dim=prot_dim,
                                 deep_projection=deep_projection))

    with tf.variable_scope('second_projection'):
        seq_refs_modified = tf.transpose(seq_refs, perm=[0, 3, 1, 2])  # (1, None, l, PROTEIN_BOW_DIM)
        if not deep_projection:
            aa_proj_ref = tf.get_variable(
                "AA_Ref_2",
                shape=(prot_dim, k),
                initializer=tf.contrib.layers.xavier_initializer())
            aa_proj_target = tf.get_variable(
                "AA_Target_2",
                shape=(prot_dim, k),
                initializer=tf.contrib.layers.xavier_initializer())

            seq_refs_projected = tf.matmul(seq_refs_modified,
                                           aa_proj_ref)  # (1, None, l, k)

            target_projected = tf.matmul(seq_target, aa_proj_target)
        else:
            seq_refs_projected = _projection_op(seq_refs_modified, n=N_LAYERS_PROJ, k=k, name='AA_Ref_2')
            target_projected = _projection_op(seq_target, n=N_LAYERS_PROJ, k=k, name='AA_Target_2')
    target_projected = tf.transpose(target_projected, perm=[0, 2, 1])  # (1, k, l)

    sequence_distance_matrices = tf.transpose(tf.matmul(
        seq_refs_projected, target_projected),
        perm=[0, 2, 3,
              1])  # (1, l, l, None)

    weighted_structures = sequence_distance_matrices + tf.concat(
        structures_info, axis=3)  # (1, l, l, None)

    return deep_conv_op(conv_input_tensor=weighted_structures,
                        **conv_params,
                        name_prefix='final')


def multi_structures_op_2(dms, seq_refs, seq_target, evfold, conv_params, ccmpred=None, k=30, prot_dim=PROTEIN_BOW_DIM):
    """Multi structures contact map prediction operation

    Args:
        dms (tf.Tensor): reference distance matrix of shape (1, l, l, None)
        seq_refs (tf.Tensor): reference sequence of shape (1, l, PROTEIN_BOW_DIM, None)
        seq_target (tf.Tensor): target sequence of shape (1, l , PROTEIN_BOW_DIM)
        evfold (tf.Tensor): plmc energy matrix of shape (1, l, l, 1)
        conv_params (dict): parameters to the last deep conv layer
        k (int): dimension to project the sequence to
        prot_dim (int): dimension of the protein numeric representation

    Returns:
        tf.Tensor: predicted contact map of shape (1, l,  l, num_bins)

    """
    structures_info = []

    for i in range(dms.get_shape().as_list()[3]):
        structures_info.append(
            _single_structure_op(dm=dms[..., i:i + 1],
                                 seq_ref=seq_refs[..., i],
                                 seq_target=seq_target,
                                 evfold=evfold,
                                 ccmpred=ccmpred,
                                 k=k,
                                 prot_dim=prot_dim,
                                 output_dim=4))

    # with tf.variable_scope('second_projection'):
    #     aa_proj_ref = tf.get_variable(
    #         "AA_Ref",
    #         shape=(prot_dim, k),
    #         initializer=tf.contrib.layers.xavier_initializer())
    #     aa_proj_target = tf.get_variable(
    #         "AA_Target",
    #         shape=(prot_dim, k),
    #         initializer=tf.contrib.layers.xavier_initializer())

    # seq_refs_modified = tf.transpose(seq_refs,
    #                                  perm=[0, 3, 1,
    #                                        2])  # (1, None, l, PROTEIN_BOW_DIM)
    # seq_refs_projected = tf.matmul(seq_refs_modified,
    #                                aa_proj_ref)  # (1, None, l, k)
    #
    # target_projected = tf.transpose(tf.matmul(seq_target, aa_proj_target),
    #                                 perm=[0, 2, 1])  # (1, k, l)
    #
    # sequence_distance_matrices = tf.transpose(tf.matmul(
    #     seq_refs_projected, target_projected),
    #     perm=[0, 2, 3,
    #           1])  # (1, l, l, None)

    weighted_structures = tf.concat(structures_info, axis=3)  # (1, l, l, None)

    return deep_conv_op(conv_input_tensor=weighted_structures,
                        **conv_params,
                        name_prefix='final')


def _projection_op(x, k, n, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        for i in range(n):
            x = tf.layers.dense(x, k, tf.nn.relu, name=f'{name}_{i}')

        return x


def _single_structure_op(dm,
                         seq_ref,
                         seq_target,
                         evfold,
                         ccmpred,
                         k,
                         name='structure_op',
                         prot_dim=PROTEIN_BOW_DIM,
                         output_dim=1,
                         deep_projection=False):
    """Performs a deep operation on a single structure to obtain a tensor to be integrated with other structures

    Args:
        dm (tf.Tensor): reference distance matrix of shape (1, l, l, 1)
        seq_ref (tf.Tensor): reference sequence of shape (1, l, PROTEIN_BOW_DIM)
        seq_target (tf.Tensor): target sequence of shape (1, l , PROTEIN_BOW_DIM)
        evfold (tf.Tensor): evfold matrix of shape (1, l, l, 1)
        ccmpred (tf.Tensor): ccmpred matrix of shape (1, l, l, 1)
        k (int): dimension to project the sequence to
        output_dim (int): last dimension of the output

    Returns:
        tf.Tensor: of shape (1, l, l, output_dim)

    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        if not deep_projection:
            aa_proj_ref = tf.get_variable(
                "AA_Ref",
                shape=(prot_dim, k),
                initializer=tf.contrib.layers.xavier_initializer())
            aa_proj_target = tf.get_variable(
                "AA_Target",
                shape=(prot_dim, k),
                initializer=tf.contrib.layers.xavier_initializer())

            p_ref = tf.matmul(seq_ref, aa_proj_ref)  # (1, l, k)
            p_target = tf.matmul(seq_target, aa_proj_target)  # (1, l, k)
        else:
            p_ref = _projection_op(seq_ref, n=N_LAYERS_PROJ, k=k, name='AA_Ref')
            p_target = _projection_op(seq_target, n=N_LAYERS_PROJ, k=k, name='AA_Target')

    distance = tf.expand_dims(tf.matmul(p_ref,
                                        tf.transpose(p_target, perm=[0, 2,
                                                                     1])),
                              axis=3)  # (1, l, l, 1)

    evo_coupling_arr = evfold if ccmpred is None else tf.concat([evfold, ccmpred], axis=3)

    context = tf.concat([distance, dm, evo_coupling_arr], axis=3)  # (1, l, l, 3) or (1, l, l, 4)

    return deep_conv_op(context,
                        num_bins=output_dim,
                        name_prefix='structure',
                        num_layers=10,
                        filter_shape=(6, 6),
                        num_channels=5)


def _project_aa_dim(input_tensor):
    """

    Args:
        input_tensor (tf.Tensor): input of shape (1 , num_homologous , l, PROTEIN_BOW_DIM)

    Returns:

    """
    # project the sparse amino acid vector to one dimension

    aa_proj_w = tf.get_variable(
        "AA_W",
        shape=(PROTEIN_BOW_DIM, 1),
        initializer=tf.contrib.layers.xavier_initializer())
    aln_aa_projection = tf.squeeze(tf.matmul(input_tensor, aa_proj_w),
                                   axis=3)  # (1, n, l)
    return aln_aa_projection


def _attention_op(input_tensor, q_dim, v_dim, num_homologous, name):
    """Performs attention operation on msa with fixed number of homologous

    Args:
        input_tensor (tf.Tensor): input of shape (1 , num_homologous , l)
        q_dim (int): dimension of queries and keys projections
        v_dim (int): dimension of values projection
        num_homologous (int): number of homologous in the alignment
        name (str): op name


    Returns:
        tf.Tensor: of shape (1 , v_dim , l)

    """

    with tf.variable_scope(name):
        w_q = tf.get_variable(
            "W_q",
            shape=(1, q_dim, num_homologous),
            initializer=tf.contrib.layers.xavier_initializer())
        w_k = tf.get_variable(
            "W_k",
            shape=(1, q_dim, num_homologous),
            initializer=tf.contrib.layers.xavier_initializer())
        w_v = tf.get_variable(
            "W_v",
            shape=(1, v_dim, num_homologous),
            initializer=tf.contrib.layers.xavier_initializer())

    q = tf.matmul(w_q, input_tensor)[0, ...]  # (q_dim, l)
    k = tf.matmul(w_k, input_tensor)[0, ...]  # (q_dim, l)
    v = tf.matmul(w_v, input_tensor)[0, ...]  # (v_dim , l)

    # l = tf.cast(input_tensor.shape[2], tf.float32)

    energy = tf.matmul(tf.transpose(q), k) / tf.sqrt(
        tf.cast(num_homologous, tf.float32))  # (l,l)

    energy_norm = tf.nn.softmax(energy, axis=1)  # (l, l)
    context = tf.matmul(v, energy_norm)  # (v_dim, l)

    return tf.nn.tanh(context)[None, ...]


def _residual_block(structure, previous_layer, num_features, num_filters,
                    name):
    """Single residual conv operation

    Args:
        structure (tf.Tensor): input of shape (1 , l , l , num_features)
        previous_layer (Union[tf.Tensor, None]): previous layer output of shape (1 , l , l , num_features)
        num_features (int): number of conv_features to input tensor
        num_filters (int): number of filters in the conv layers
        name (str): name for the residual operation

    Returns:
        tf.Tensor: same shape as structure

    """
    previous_layer = previous_layer if previous_layer is not None else tf.zeros_like(
        structure)

    with tf.variable_scope(name):
        residual_sum = structure + previous_layer

        conv1 = pairwise_conv_layer(residual_sum,
                                    num_features=num_features,
                                    num_filters=num_filters,
                                    filter_shape=(5, 5),
                                    dilation=1,
                                    name='conv1')

        conv2 = pairwise_conv_layer(conv1,
                                    num_features=num_filters,
                                    num_filters=num_features,
                                    filter_shape=(7, 7),
                                    dilation=1,
                                    name='conv2')

        return conv2


def residual_structures_op(references_tensor, conv_input_tensor, conv_params,
                           shared_weights, num_res_filters):
    """Residual structure operation

    Args:
        references_tensor (tf.Tensor): input of shape (1 , l , l , num_features, num_refrences)
        conv_input_tensor (tf.Tensor): input to conv net if shape (1, l , l , Any)
        conv_params (dict): params to last conv op
        shared_weights (bool): if true weights are shared among layers
        num_res_filters (int): number of filters in the residual block

    Returns:
        tf.Tensor: predicted contact map of shape 1 * l * l * num_bins

    """
    previous_layer = None
    num_features = references_tensor.get_shape().as_list()[3]
    num_refrences = references_tensor.get_shape().as_list()[-1]

    for i in range(num_refrences):
        structure = references_tensor[..., i]
        is_zero = tf.equal(tf.reduce_sum(structure), 0)

        name = 'res_block_%s' % i if shared_weights else 'res_block'

        res_op = _residual_block(structure=structure,
                                 previous_layer=previous_layer,
                                 num_features=num_features,
                                 num_filters=num_res_filters,
                                 name=name)

        res_op = tf.where(is_zero, structure, res_op)

        previous_layer = res_op

    conv_combined_input = tf.concat([conv_input_tensor, previous_layer],
                                    axis=3)

    return deep_conv_op(conv_combined_input, **conv_params)


def deep_attention_op(attention_input_tensor, conv_input_tensor, num_layers,
                      q_dim, v_dim, conv_parmas):
    """Performs deep attention operation

    Args:
        attention_input_tensor (tf.Tensor): input of shape (1 , num_homologous , l, PROTEIN_BOW_DIM)
        conv_input_tensor (tf.Tensor): input of shape (1 , l , l , input_shape)
        num_layers (int): number of hidden layers
        q_dim (int): damnation of queries and keys projections
        v_dim (int): dimension of values projection
        conv_parmas (dict): inputs to :meth: `deep_conv_op`


    Returns:
        tf.Tensor: tensor of shape (1, l, l, 1)

    """

    projected_aa = _project_aa_dim(attention_input_tensor)
    num_homologous = projected_aa.get_shape().as_list()[1]

    current_attention = _attention_op(input_tensor=projected_aa,
                                      q_dim=q_dim,
                                      v_dim=v_dim,
                                      num_homologous=num_homologous,
                                      name='attention_%s' % 0)

    for i in range(1, num_layers):
        prev_attention = current_attention

        current_attention = _attention_op(input_tensor=prev_attention,
                                          q_dim=q_dim,
                                          v_dim=v_dim,
                                          num_homologous=v_dim,
                                          name='attention_%s' % i)

    outer_product = []

    for i in range(v_dim):
        outer_product.append(
            tf.tensordot(current_attention[:, i, :],
                         tf.transpose(current_attention[:, i, :]),
                         axes=0))

    attention_pairwise_mat = tf.concat(outer_product, axis=3)

    conv_combined_input = tf.concat(
        [attention_pairwise_mat, conv_input_tensor], axis=3)

    return deep_conv_op(conv_combined_input, **conv_parmas)


def deep_conv_op(conv_input_tensor,
                 num_layers=6,
                 num_channels=5,
                 filter_shape=(5, 5),
                 dilation=1,
                 num_bins=1,
                 residual=False,
                 name_prefix=None):
    """Produces predicted contact map

    Args:
        conv_input_tensor (tf.Tensor): input of shape (1 , l , l , input_shape)
        num_layers (int): number of hidden layers
        num_channels (int): number of channels for each hidden layer
        filter_shape (Union[tuple, list[tuple]): shape of the convolution filter
        dilation (Union[int, list[int]): dilation factor for conv layer
        num_bins (int): number of bins to predicts, if 1 it's a regression predictor
        residual (bool): if true we use 2d conv resnet
        name_prefix (str): prefix for conv scope name
    Returns:
        tf.Tensor: predicted contact map of shape (1, l,  l, num_bins)

    """

    def _extend_param_to_layers(param, n_layers):
        if type(param) != list:
            param = [param]
        if len(param) == n_layers:
            return param
        else:
            return param * n_layers

    filters = _extend_param_to_layers(filter_shape, num_layers + 2)
    dilations = _extend_param_to_layers(dilation, num_layers + 2)

    input_shape = int(conv_input_tensor.shape[-1])

    if len(conv_input_tensor.shape) == 3:
        conv_input_tensor = tf.expand_dims(conv_input_tensor, axis=0)

    conv_name = '%s_conv_input' % name_prefix if name_prefix is not None else 'conv_input'
    conv = pairwise_conv_layer(
        input_data=conv_input_tensor,
        num_features=input_shape,
        num_filters=num_channels,
        filter_shape=filters[0],
        dilation=dilations[0],
        name=conv_name,
    )
    previous_layer = conv
    for i in range(0, num_layers):
        conv_name_i = '%s_conv_h_%s' % (
            name_prefix, i) if name_prefix is not None else 'conv_h_%s' % i
        conv = pairwise_conv_layer(input_data=previous_layer,
                                   num_features=num_channels,
                                   num_filters=num_channels,
                                   filter_shape=filters[i + 1],
                                   dilation=dilations[i + 1],
                                   name=conv_name_i,
                                   residual=residual)

        previous_layer = conv

    def _define_last_layer(num_bins, previous_layer, num_channels,
                           filter_shape, dilation, C_MAP_PRED):
        if num_bins == 1:
            last_name = '%s_last_conv' % name_prefix if name_prefix is not None else 'last_conv'
            last_conv = pairwise_conv_layer(input_data=previous_layer,
                                            num_features=num_channels,
                                            num_filters=num_bins,
                                            dilation=dilation,
                                            filter_shape=filter_shape,
                                            name=last_name)
            last_layer = tf.identity(
                last_conv + tf.transpose(last_conv, perm=[0, 2, 1, 3]),
                name=C_MAP_PRED)
        elif num_bins > 1:
            last_name = '%s_conv_h_%s' % (
                name_prefix, num_layers
            ) if name_prefix is not None else 'conv_h_%s' % num_layers
            conv = pairwise_conv_layer(input_data=previous_layer,
                                       num_features=num_channels,
                                       num_filters=num_bins,
                                       filter_shape=filter_shape,
                                       dilation=dilation,
                                       name=last_name)
            last_layer = tf.nn.softmax(conv +
                                       tf.transpose(conv, perm=[0, 2, 1, 3]),
                                       axis=-1,
                                       name=C_MAP_PRED)

        return last_layer

    contact_predictions = _define_last_layer(num_bins, previous_layer,
                                             num_channels, filters[-1],
                                             dilations[-1], C_MAP_PRED)

    return contact_predictions


def get_opt_op(loss, global_step, lr=0.0001):
    """Returns the optimzer over a given loss

    Args:
        loss (tf.Tensor): the loss tensor
        global_step (tf.Tensor): the global step
        lr (float): the optimization learning rate

    Returns:
        tf.Tensor: optimization tensor

    """

    opt_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(
        loss, global_step=global_step)
    return opt_op
