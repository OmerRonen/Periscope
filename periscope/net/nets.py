import tensorflow as tf

from periscope.net.basic_ops import (pairwise_conv_layer_2, deep_conv_op, pairwise_conv_layer, PROJECTION_DIM,
                                     outer_concat, print_max_min)


def _weighting_op_template(seq_template, seq_target, conv_layer=pairwise_conv_layer):
    """Computes weights for template

    Args:
        seq_template (tf.Tensor): template sequence/pwm of shape (1, l, 22/21)
        seq_target (tf.Tensor): target sequence/conservation of shape (1, l, 22/21)
        conv_layer (function): conv layer to use for deep conv

    Returns:
        tf.Tensor: weights of shape (1, l, l, 1)

    """
    with tf.variable_scope('weight_template_op', reuse=tf.AUTO_REUSE):
        seq_template = seq_template[..., 0:21]
        aa_dim = tf.shape(seq_template)[-1]
        aa_dim_int = int(seq_template.shape[-1])
        l = tf.shape(seq_template)[1]
        one = tf.shape(seq_template)[0]

        name_w = 'template'

        O = tf.einsum('ijk,ijl->ijkl', seq_template, seq_target)
        O_s = tf.reshape(O, shape=(one, l, aa_dim * aa_dim))
        conv0_filt_shape = [
            10, aa_dim_int ** 2, PROJECTION_DIM
        ]
        # initialise weights and bias for the filter
        weights0 = tf.get_variable(
            f"{name_w}_conv0_w",
            shape=conv0_filt_shape,
            initializer=tf.contrib.layers.xavier_initializer())
        # weights0 = _print_max_min(weights0, "weights0 template")
        # O_s = _print_max_min(O_s, "O_s template")

        O_smooth = tf.nn.tanh(tf.nn.conv1d(input=O_s, filters=weights0, padding='SAME'))
        # O_smooth = _print_max_min(O_smooth, "O_smooth template")

        conv0_filt_shape = [
            10, PROJECTION_DIM, 1
        ]
        # initialise weights and bias for the filter
        weights1 = tf.get_variable(
            f"{name_w}_conv1_w",
            shape=conv0_filt_shape,
            initializer=tf.contrib.layers.xavier_initializer())
        # weights1 = _print_max_min(weights1, "weights1 template")

        W_1d = tf.nn.conv1d(input=O_smooth, filters=weights1, padding='SAME')
        W = outer_concat(W_1d)
        input_shape = 2
        W_smooth = deep_conv_op(W, input_shape=input_shape, name_prefix=name_w, num_bins=1, residual=True,
                                num_layers=2, conv_layer=conv_layer)

        return W_smooth


def _weighting_op_evo(conservation, evo_arr, beff, name, conv_layer=pairwise_conv_layer):
    """Computes weights for evo

    Args:
        conservation (tf.Tensor): conservation score  of shape (1, l, 1)
        evo_arr (tf.Tensor): ccmpred/evfold energy matrix of shape (1, l, l, 1)
        beff (tf.Tensor): Effective number of sequences within alignment Beff
        name (str): layer name
        conv_layer (function): conv layer for deep conv


    Returns:
        tf.Tensor: weights of shape (1, l, l, 1)

    """
    with tf.variable_scope('weight_evo_op', reuse=tf.AUTO_REUSE):
        conservation_outer = outer_concat(conservation)
        # conservation_outer = _print_max_min(conservation_outer, "conservation_outer ")

        shape_w = tf.shape(conservation_outer)
        beff_arr = tf.ones(shape=(shape_w[0], shape_w[1], shape_w[2], shape_w[0])) * tf.log(beff)
        # beff_arr = _print_max_min(beff_arr, "beff_arr ")

        W = tf.concat([beff_arr, evo_arr], axis=3)

        # W = _print_max_min(W, f"W {name} ")

        W_smooth = deep_conv_op(W, input_shape=2, name_prefix=name, num_bins=1, residual=True, num_layers=2,
                                conv_layer=conv_layer)
        # W_smooth = _print_max_min(W_smooth, f"W_smooth evo {name}")

        return W_smooth


def _template_op(seq, dm, conv_layer=pairwise_conv_layer):
    """Performs Template network operation

    Args:
        seq (tf.Tensor): reference sequence of shape (1, l, 22)
        dm (tf.Tensor): reference distance matrix of shape (1, l, l, 1)
        conv_layer (function): the conv layer to use in deep conv


    Returns:
        tf.Tensor: contact map embedding of shape (1, l, l, 1)

    """
    with tf.variable_scope('template_op', reuse=tf.AUTO_REUSE):
        seq_dim = int(seq.shape[-1])
        aa_proj_ref = tf.get_variable(
            "ref_projection",
            shape=(seq_dim, PROJECTION_DIM),
            initializer=tf.contrib.layers.xavier_initializer())
        # aa_proj_ref = _print_max_min(aa_proj_ref, "aa_proj_ref template op")
        s = tf.matmul(seq, aa_proj_ref)  # (1, l, PROJECTION_DIM)
        s_pair = outer_concat(s)
        c = tf.concat([s_pair, dm], axis=3)
        logits = deep_conv_op(c, input_shape=PROJECTION_DIM * 2 + 1, num_bins=2, name_prefix='template_op_conv',
                              residual=True, conv_layer=conv_layer)

        logits = logits[..., 1:]
        # logits = _print_max_min(logits, "logits template op")
        return logits


def _evo_op(pwn, evo_arr, conv_layer=pairwise_conv_layer):
    """Performs Template network operation

    Args:
        pwn (tf.Tensor): pssm of shape (1, l, 21)
        evo_arr (tf.Tensor): evfold/ccmpred matrix of shape (1, l, l, 1)
        conv_layer (function): conv layer for deep conv


    Returns:
        tf.Tensor: contact map embedding of shape (1, l, l, 1)

    """
    shp = pwn.get_shape().as_list()[-1]
    with tf.variable_scope('evo_op', reuse=tf.AUTO_REUSE):
        aa_proj_ref = tf.get_variable(
            "pwn_projection",
            shape=(shp, PROJECTION_DIM),
            initializer=tf.contrib.layers.xavier_initializer())
        # aa_proj_ref = _print_max_min(aa_proj_ref, 'aa_proj_ref')
        s = tf.matmul(pwn, aa_proj_ref)  # (1, l, PROJECTION_DIM)
        s_pair = outer_concat(s)
        c = tf.concat([s_pair, evo_arr], axis=3)
        logits = deep_conv_op(c, input_shape=PROJECTION_DIM * 2 + 1, num_bins=2, name_prefix='evo_op_conv',
                              residual=True, conv_layer=conv_layer)
        logits = logits[..., 1:]  # * tf.constant(float(2))
        # logits = _print_max_min(logits, "logits evo op")
        return logits


def _target_property_op(seq_target, properties_target, conv_layer=pairwise_conv_layer_2):
    """Computes weights for target property

    Args:
        seq_target (tf.Tensor): target sequence of shape (1, l, 22/21)
        properties_target (tf.Tensor): target properties of shape (1, l, 13)
        conv_layer (function): conv layer to use for deep conv

    Returns:
        tf.Tensor: weights of shape (1, l, l, 1)

    """
    with tf.variable_scope('target_property_op', reuse=tf.AUTO_REUSE):
        seq_target = seq_target[..., 0:21]
        aa_dim_int = int(seq_target.shape[-1]) + int(properties_target.shape[-1])

        name_w = 'target_property'

        O_s = tf.concat([seq_target, properties_target], axis=2)
        conv0_filt_shape = [
            10, aa_dim_int, PROJECTION_DIM
        ]
        # initialise weights and bias for the filter
        weights0 = tf.get_variable(
            f"{name_w}_conv0_w",
            shape=conv0_filt_shape,
            initializer=tf.contrib.layers.xavier_initializer())
        # weights0 = _print_max_min(weights0, "weights0 template")
        # O_s = print_max_min(O_s, "O_s property")

        O_smooth = tf.nn.tanh(tf.nn.conv1d(input=O_s, filters=weights0, padding='SAME'))
        # O_smooth = print_max_min(O_smooth, "O_smooth property")

        conv0_filt_shape = [
            10, PROJECTION_DIM, 1
        ]
        # initialise weights and bias for the filter
        weights1 = tf.get_variable(
            f"{name_w}_conv1_w",
            shape=conv0_filt_shape,
            initializer=tf.contrib.layers.xavier_initializer())
        # weights1 = print_max_min(weights1, "weights1 property")

        W_1d = tf.nn.conv1d(input=O_smooth, filters=weights1, padding='SAME')
        W = outer_concat(W_1d)
        # W = print_max_min(W, "W property")

        input_shape = 2
        W_smooth = deep_conv_op(W, input_shape=input_shape, name_prefix=name_w, num_bins=1, residual=True,
                                num_layers=2, conv_layer=conv_layer)
        # W_smooth = print_max_min(W_smooth, "W_smooth property")

        return W_smooth


def periscope_net_properties(dms, seq_refs, seq_target, properties_target, ccmpred, pwm_w, pwm_evo, conservation, beff,
                             conv_params):
    """Neural network that predicts a protein contact map

    Args:
        dms (tf.Tensor): reference distance matrix of shape (1, l, l, None)
        seq_refs (tf.Tensor): reference sequence of shape (1, l, 22, None)
        seq_target (tf.Tensor): target sequence of shape (1, l, 22, None)
        properties_target (tf.Tensor): target properties of shape (1, l, 13, None)
        pwm_w (tf.Tensor): position specific weight matrix focused on the wild type of shape (1, l, 21)
        pwm_evo (tf.Tensor): position specific weight averaged across the entire alignment matrix of shape (1, l, 21)
        conservation (tf.Tensor): conservation score tensor of shape (1, l, 21)
        beff (tf.Tensor): Effective number of sequences within alignment Beff
        conv_params (dict): parameters to the last deep conv layer


    Returns:
        tf.Tensor: predicted contact map of shape (1, l,  l, num_bins)

    """
    with tf.variable_scope('periscope', reuse=tf.AUTO_REUSE):
        weights = []
        templates = []
        for i in range(dms.get_shape().as_list()[3]):
            dm = dms[..., i:i + 1]
            seq = seq_refs[..., i]
            max_dm = tf.reduce_max(dm) * tf.ones_like(dm)
            t = tf.where(max_dm > 0, _template_op(dm=dm, seq=seq, conv_layer=pairwise_conv_layer_2), dm)
            w = tf.where(max_dm > 0, _weighting_op_template(seq_template=seq, seq_target=pwm_w,
                                                            conv_layer=pairwise_conv_layer_2), dm)

            templates.append(t)
            weights.append(w)

        templates += [_evo_op(pwm_evo, ccmpred, conv_layer=pairwise_conv_layer_2)]  # , _evo_op(pwm_evo, evfold)]
        w_ccmpred = _weighting_op_evo(conservation, beff=beff, evo_arr=ccmpred, name='ccmpred',
                                      conv_layer=pairwise_conv_layer_2)

        weights += [w_ccmpred]

        templates = tf.concat(templates, axis=3)
        # templates = _print_max_min(templates, "concat templates")

        weights = tf.concat(weights, axis=3)
        # weights = _print_max_min(weights, "concat weights")

        # weights = _print_max_min(weights, "softmax weights")
        K = dms.get_shape().as_list()[3]
        max_dm = tf.reduce_max(tf.reduce_max(dms, axis=1), axis=1)
        N_templates = tf.reduce_sum(tf.where(max_dm > 0, tf.ones_like(max_dm), tf.zeros_like(max_dm)))
        factor_templates = tf.ones_like(weights[..., :K]) * tf.log(tf.reduce_max([N_templates, float(1)]))
        factor_evo = tf.ones_like(weights[..., K:]) * tf.log(float(1))
        factor = tf.concat([factor_templates, factor_evo], axis=3)
        # factor = _print_max_min(factor, 'factor')
        weights = tf.nn.softmax(weights - factor, axis=3)

        weighted_tempaltes = templates * weights
        target_property = _target_property_op(seq_target, properties_target)
        conv_inp = tf.concat([tf.reduce_sum(weighted_tempaltes, axis=3, keepdims=True), target_property], axis=3)

        cmap_hat = deep_conv_op(conv_inp, **conv_params, name_prefix='final',
                                residual=True, conv_layer=pairwise_conv_layer_2)
        return cmap_hat, weights


def periscope_net(dms, seq_refs, ccmpred, pwm_w, pwm_evo, conservation, beff, conv_params):
    """Neural network that predicts a protein contact map

    Args:
        dms (tf.Tensor): reference distance matrix of shape (1, l, l, None)
        seq_refs (tf.Tensor): reference sequence of shape (1, l, 22, None)
        evfold (tf.Tensor): evfold energy matrix of shape (1, l, l, 1)
        pwm_w (tf.Tensor): position specific weight matrix focused on the wild type of shape (1, l, 21)
        pwm_evo (tf.Tensor): position specific weight averaged across the entire alignment matrix of shape (1, l, 21)
        conservation (tf.Tensor): conservation score tensor of shape (1, l, 21)
        beff (tf.Tensor): Effective number of sequences within alignment Beff
        conv_params (dict): parameters to the last deep conv layer


    Returns:
        tf.Tensor: predicted contact map of shape (1, l,  l, num_bins)

    """
    with tf.variable_scope('periscope', reuse=tf.AUTO_REUSE):
        weights = []
        templates = []
        for i in range(dms.get_shape().as_list()[3]):
            dm = dms[..., i:i + 1]
            seq = seq_refs[..., i]
            max_dm = tf.reduce_max(dm) * tf.ones_like(dm)
            t = tf.where(max_dm > 0, _template_op(dm=dm, seq=seq, conv_layer=pairwise_conv_layer_2), dm)
            w = tf.where(max_dm > 0, _weighting_op_template(seq_template=seq, seq_target=pwm_w,
                                                            conv_layer=pairwise_conv_layer_2), dm)

            templates.append(t)
            weights.append(w)

        templates += [_evo_op(pwm_evo, ccmpred, conv_layer=pairwise_conv_layer_2)]  # , _evo_op(pwm_evo, evfold)]
        w_ccmpred = _weighting_op_evo(conservation, beff=beff, evo_arr=ccmpred, name='ccmpred',
                                      conv_layer=pairwise_conv_layer_2)

        weights += [w_ccmpred]

        templates = tf.concat(templates, axis=3)
        # templates = _print_max_min(templates, "concat templates")

        weights = tf.concat(weights, axis=3)
        # weights = _print_max_min(weights, "concat weights")

        # weights = _print_max_min(weights, "softmax weights")
        K = dms.get_shape().as_list()[3]
        max_dm = tf.reduce_max(tf.reduce_max(dms, axis=1), axis=1)
        N_templates = tf.reduce_sum(tf.where(max_dm > 0, tf.ones_like(max_dm), tf.zeros_like(max_dm)))
        factor_templates = tf.ones_like(weights[..., :K]) * tf.log(tf.reduce_max([N_templates, float(1)]))
        factor_evo = tf.ones_like(weights[..., K:]) * tf.log(float(1))
        factor = tf.concat([factor_templates, factor_evo], axis=3)
        # factor = _print_max_min(factor, 'factor')
        weights = tf.nn.softmax(weights - factor, axis=3)

        weighted_tempaltes = templates * weights
        conv_inp = tf.reduce_sum(weighted_tempaltes, axis=3, keepdims=True)
        cmap_hat = deep_conv_op(conv_inp, **conv_params, name_prefix='final',
                                residual=True, conv_layer=pairwise_conv_layer_2)
        return cmap_hat, weights


def template_net(dms, seq_refs, pwm_w, conv_params):
    """Neural network that predicts a protein contact map

    Args:
        dms (tf.Tensor): reference distance matrix of shape (1, l, l, None)
        seq_refs (tf.Tensor): reference sequence of shape (1, l, 22, None)
        pwm_w (tf.Tensor): position specific weight matrix focused on the wild type of shape (1, l, 21)
        conv_params (dict): parameters to the last deep conv layer


    Returns:
        tf.Tensor: predicted contact map of shape (1, l,  l, num_bins)

    """
    with tf.variable_scope('template', reuse=tf.AUTO_REUSE):
        weights = []
        templates = []
        for i in range(dms.get_shape().as_list()[3]):
            dm = dms[..., i:i + 1]
            seq = seq_refs[..., i]
            max_dm = tf.reduce_max(dm) * tf.ones_like(dm)
            t = tf.where(max_dm > 0, _template_op(dm=dm, seq=seq, conv_layer=pairwise_conv_layer_2), dm)
            w = tf.where(max_dm > 0,
                         _weighting_op_template(seq_template=seq, seq_target=pwm_w, conv_layer=pairwise_conv_layer_2),
                         dm)

            templates.append(t)
            weights.append(w)

        templates = tf.concat(templates, axis=3)

        weights = tf.concat(weights, axis=3)

        weights = tf.nn.softmax(
            weights,
            axis=3)

        weighted_tempaltes = templates * weights
        # weighted_tempaltes = _print_max_min(weighted_tempaltes, "weighted_tempaltes")

        conv_inp = tf.reduce_sum(weighted_tempaltes, axis=3, keepdims=True)
        # conv_inp = _print_max_min(conv_inp, "conv_inp")
        cmap_hat = deep_conv_op(conv_inp, **conv_params, name_prefix='final',
                                residual=True, conv_layer=pairwise_conv_layer_2)
        return cmap_hat, weights


def evo_net(evfold, ccmpred, pwm_evo, conservation, beff, conv_params):
    """Neural network that predicts a protein contact map

    Args:
        evfold (tf.Tensor): evfold energy matrix of shape (1, l, l, 1)
        ccmpred (tf.Tensor): ccmpred energy matrix of shape (1, l, l, 1)
        pwm_evo (tf.Tensor): position specific weight averaged across the entire alignment matrix of shape (1, l, 21)
        conservation (tf.Tensor): conservation score tensor of shape (1, l, 21)
        beff (tf.Tensor): Effective number of sequences within alignment Beff
        conv_params (dict): parameters to the last deep conv layer


    Returns:
        tf.Tensor: predicted contact map of shape (1, l,  l, num_bins)

    """
    with tf.variable_scope('template', reuse=tf.AUTO_REUSE):
        weights = []

        evo = [_evo_op(pwm_evo, ccmpred, pairwise_conv_layer_2), _evo_op(pwm_evo, evfold, pairwise_conv_layer_2)]
        w_ccmpred = _weighting_op_evo(conservation, beff=beff, evo_arr=ccmpred, name='ccmpred',
                                      conv_layer=pairwise_conv_layer_2)
        w_evfold = _weighting_op_evo(conservation, beff=beff, evo_arr=evfold, name='evfold',
                                     conv_layer=pairwise_conv_layer_2)
        weights += [w_ccmpred, w_evfold]

        evo = tf.concat(evo, axis=3)

        weights = tf.concat(weights, axis=3)

        weights = tf.nn.softmax(
            weights,
            axis=3)

        weighted_evo = evo * weights

        conv_inp = tf.reduce_sum(weighted_evo, axis=3, keepdims=True)
        cmap_hat = deep_conv_op(conv_inp, **conv_params, name_prefix='final',
                                residual=True, conv_layer=pairwise_conv_layer_2)
        return cmap_hat, weights
