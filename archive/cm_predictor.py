import os

import logging
import random
from functools import partial

import tensorflow as tf
import numpy as np

from .data_handler import ProteinDataHandler
from .globals import FEATURES_DIMS, PREDICTON_FEATURES, DATASETS, ARCHS, FEATURES, PROTEIN_BOW_DIM, PROTEIN_BOW_DIM_PSSM, \
    PROTEIN_BOW_DIM_PSSM_SS, PROTEIN_BOW_DIM_SS
from .protein_net import (deep_conv_op, upper_triangular_mse_loss,
                         upper_triangular_cross_entropy_loss, get_opt_op,
                         get_top_category_accuracy, compare_predictions,
                         residual_structures_op, multi_structures_op, multi_structures_op_2)
from .utils_old import pkl_save, yaml_save

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)


def check_create_folder(folder_path):
    if os.path.exists(folder_path):
        return
    os.mkdir(folder_path)
    LOGGER.info('%s created' % folder_path)


LOSS_FUNCTIONS = {
    'regression': upper_triangular_mse_loss,
    'classification': upper_triangular_cross_entropy_loss
}
logging.basicConfig(level=logging.INFO)

LOGGER = logging.getLogger(__name__)


class ProteinNet:
    """neural net that outputs predicted contact map"""

    def __init__(self, net_params):
        BIN_THRESHOLDS = {1: 8.5, 2: 1}

        self.arch = net_params['architecture']['arch']

        self._conv_arch_params = net_params['architecture']['conv']
        self._resnet_arch_params = net_params['architecture'][
            'references_resnet']

        self._data_params = net_params['data']
        self._features = self._data_params['conv_features']
        self.num_bins = self._conv_arch_params['num_bins']
        self._dilation = self._conv_arch_params['dilation']
        self.output_shape = (None, None, self.num_bins)

        self.prediction_type = 'regression' if self.num_bins == 1 else 'classification'
        self.threshold = BIN_THRESHOLDS[self.num_bins]

    def get_top_prediction(self, predictions, contact_map, sequence_length,
                           mode):
        top_l_predictions = get_top_category_accuracy(
            category='L',
            top=1,
            predictions=predictions,
            contact_map=contact_map,
            sequence_length=sequence_length,
            mode=mode)
        top_m_predictions = get_top_category_accuracy(
            category='M',
            top=1,
            predictions=predictions,
            contact_map=contact_map,
            sequence_length=sequence_length,
            mode=mode)
        top_s_predictions = get_top_category_accuracy(
            category='S',
            top=1,
            predictions=predictions,
            contact_map=contact_map,
            sequence_length=sequence_length,
            mode=mode)
        top_predictions = {
            'top_l_contacts/long_range': top_l_predictions,
            'top_l_contacts/medium_range': top_m_predictions,
            'top_l_contacts/short_range': top_s_predictions
        }
        return top_predictions

    @property
    def predicted_contact_map(self):
        if self.arch == ARCHS.conv:
            return partial(deep_conv_op, **self._conv_arch_params)

        elif self.arch == ARCHS.references_resnet:
            return partial(residual_structures_op,
                           conv_params=self._conv_arch_params,
                           **self._resnet_arch_params)

        elif self.arch == ARCHS.multi_structure or self.arch == ARCHS.multi_structure_ccmpred:
            return partial(multi_structures_op,
                           conv_params=self._conv_arch_params,
                           prot_dim=PROTEIN_BOW_DIM)

        elif self.arch == ARCHS.multi_structure_ccmpred_2:
            return partial(multi_structures_op_2,
                           conv_params=self._conv_arch_params,
                           prot_dim=PROTEIN_BOW_DIM)

        elif self.arch == ARCHS.multi_structure_pssm:
            return partial(multi_structures_op,
                           conv_params=self._conv_arch_params,
                           prot_dim=PROTEIN_BOW_DIM_PSSM,
                           k=100)

        elif self.arch == ARCHS.ms_ccmpred_pssm:
            return partial(multi_structures_op,
                           conv_params=self._conv_arch_params,
                           prot_dim=PROTEIN_BOW_DIM_PSSM,
                           k=100)
        elif self.arch == ARCHS.ms_ss_ccmpred_pssm:
            return partial(multi_structures_op_2,
                           conv_params=self._conv_arch_params,
                           prot_dim=PROTEIN_BOW_DIM_PSSM_SS,
                           k=100)
        elif self.arch == ARCHS.ms_ss_ccmpred:
            return partial(multi_structures_op,
                           conv_params=self._conv_arch_params,
                           prot_dim=PROTEIN_BOW_DIM_SS,
                           k=100)
        elif self.arch == ARCHS.ms_ss_ccmpred_2:
            return partial(multi_structures_op_2,
                           conv_params=self._conv_arch_params,
                           prot_dim=PROTEIN_BOW_DIM_SS,
                           k=100)

    def define_prediction_methods_similarity(self, predictions, inputs_dict,
                                             sequence_length, mode):
        prediction_methods_similarity = {}
        for pred_mat in PREDICTON_FEATURES.intersection(set(self._features)):
            const = -1 if 'dm' in pred_mat.split('_') else 1
            similarity = compare_predictions(predictions,
                                             const * inputs_dict[pred_mat],
                                             sequence_length, mode)
            prediction_methods_similarity['methods_similarity/%s' %
                                          pred_mat] = similarity
        return prediction_methods_similarity


class ContactMapPredictor:
    """Contact map predictor"""

    def __init__(self, params):
        self._train_params = params['train']
        self.path = self._train_params['path']
        check_create_folder(self.path)
        self.artifacts_path = os.path.join(self.path, 'artifacts')
        check_create_folder(self.artifacts_path)

        params_file = os.path.join(self.path, 'params.yml')
        if not os.path.isfile(params_file):
            yaml_save(filename=params_file, data=params)

        self._opt_params = self._train_params['opt']

        self.net = ProteinNet(params['net'])
        self._batch_size = self._train_params['batch_size']

        self.conv_features = params['net']['data']['conv_features']
        self.resnet_features = params['net']['data']['resnet_features']

        self._arch = params['net']['architecture']['arch']
        k = params['net']['data']['k']
        log_pssm = params['net']['data']['log_pssm']

        train_dataset_name = self._train_params['train_dataset']
        eval_dataset_name = self._train_params['eval_dataset']
        test_dataset_name = self._train_params['test_dataset']

        train_proteins = DATASETS.get(train_dataset_name, None)
        eval_proteins = DATASETS.get(eval_dataset_name, None)
        test_proteins = DATASETS.get(test_dataset_name, None)

        data_generator_args = {
            'arch': self._arch,
            'conv_features': self.conv_features,
            'resnet_features': self.resnet_features,
            'k': k,
            'num_bins': self.net.num_bins,
            'model_path': self.artifacts_path,
            'log_pssm': log_pssm
        }
        train_epochs = self._train_params['epochs']
        eval_epochs = 1
        test_epochs = 1

        self.train_data_manager = DataGenerator(proteins=train_proteins,
                                                mode='train',
                                                epochs=train_epochs,
                                                batch_size=self._batch_size,
                                                **data_generator_args)
        self.eval_data_manager = DataGenerator(proteins=eval_proteins,
                                               mode='eval',
                                               epochs=eval_epochs,
                                               **data_generator_args)
        self.predict_data_manager = DataGenerator(proteins=test_proteins,
                                                  mode='predict',
                                                  epochs=test_epochs,
                                                  **data_generator_args)

        gpu_options = tf.GPUOptions(allow_growth=True)
        session_config = tf.ConfigProto(gpu_options=gpu_options)
        config = tf.estimator.RunConfig(
            save_summary_steps=self._train_params['in_train']
            ['save_summary_steps'],
            save_checkpoints_steps=self._train_params['in_train']
            ['save_checkpoints_steps'],
            session_config=session_config,
            log_step_count_steps=None,
            tf_random_seed=42)

        self.estimator = tf.estimator.Estimator(self._get_model_fn(),
                                                self.path,
                                                params={},
                                                config=config)

    def _get_opt_op(self, loss, global_step):
        return get_opt_op(loss, global_step, **self._opt_params)

    def _get_metrics_dict(self, contact_pred_cm, cm, seq_len, features_dict):
        metrics_dict = self.net.get_top_prediction(contact_pred_cm, cm,
                                                   seq_len, 'eval')
        metrics_dict.update(
            self.net.define_prediction_methods_similarity(
                contact_pred_cm, features_dict, seq_len, 'eval'))

        return metrics_dict

    def _get_inputs(self, features):

        if self.net.arch == ARCHS.multi_structure:
            multi_structures_input = {
                'dms': features[FEATURES.k_reference_dm_conv],
                'seq_refs': features[FEATURES.seq_refs],
                'seq_target': features[FEATURES.seq_target],
                'plmc': features[FEATURES.plmc_score]
            }
            return multi_structures_input

        if self.net.arch == ARCHS.multi_structure_ccmpred or self._arch == ARCHS.multi_structure_ccmpred_2:
            multi_structures_input = {
                'dms': features[FEATURES.k_reference_dm_conv],
                'seq_refs': features[FEATURES.seq_refs],
                'seq_target': features[FEATURES.seq_target],
                'plmc': features[FEATURES.plmc_score],
                'ccmpred': features[FEATURES.ccmpred]
            }
            return multi_structures_input

        if self.net.arch == ARCHS.multi_structure_pssm:
            multi_structures_input = {
                'dms': features[FEATURES.k_reference_dm_conv],
                'seq_refs': features[FEATURES.seq_refs_pssm],
                'seq_target': features[FEATURES.seq_target_pssm],
                'plmc': features[FEATURES.plmc_score]
            }
            return multi_structures_input
        if self.net.arch == ARCHS.ms_ccmpred_pssm:
            multi_structures_input = {
                'dms': features[FEATURES.k_reference_dm_conv],
                'seq_refs': features[FEATURES.seq_refs_pssm],
                'seq_target': features[FEATURES.seq_target_pssm],
                'plmc': features[FEATURES.plmc_score],
                'ccmpred': features[FEATURES.ccmpred]
            }
            return multi_structures_input
        if self.net.arch == ARCHS.ms_ss_ccmpred_pssm:
            multi_structures_input = {
                'dms': features[FEATURES.k_reference_dm_conv],
                'seq_refs': features[FEATURES.seq_refs_pssm_ss],
                'seq_target': features[FEATURES.seq_target_pssm_ss],
                'plmc': features[FEATURES.plmc_score],
                'ccmpred': features[FEATURES.ccmpred]
            }
            return multi_structures_input
        if self.net.arch == ARCHS.ms_ss_ccmpred or self.net.arch == ARCHS.ms_ss_ccmpred_2:
            multi_structures_input = {
                'dms': features[FEATURES.k_reference_dm_conv],
                'seq_refs': features[FEATURES.seq_refs_ss],
                'seq_target': features[FEATURES.seq_target_ss],
                'plmc': features[FEATURES.plmc_score],
                'ccmpred': features[FEATURES.ccmpred]
            }
            return multi_structures_input

        features_dict = {
            f: d
            for f, d in features.items() if f in self.conv_features
        }
        inputs = {
            'conv_input_tensor': tf.concat(list(features_dict.values()),
                                           axis=3)
        }

        if self.net.arch == ARCHS.conv:
            return inputs

        elif self.net.arch == ARCHS.references_resnet:

            resnet_inputs = {
                'conv_input_tensor': inputs['conv_input_tensor'],
                'references_tensor': features[self.resnet_features[0]]
            }

            return resnet_inputs

    def _get_model_fn(self):
        def model_fn(features, labels, mode, params):

            inputs = self._get_inputs(features)

            contact_pred = self.net.predicted_contact_map(**inputs)

            max = tf.reduce_max(contact_pred)

            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(mode,
                                                  predictions=contact_pred)

            cm = features['contact_map']

            seq_len = features['sequence_length'][0, 0]

            loss_fn = LOSS_FUNCTIONS[self.net.prediction_type]
            loss = loss_fn(contact_pred, cm)
            global_step = tf.train.get_global_step()

            if self.net.prediction_type == 'classification':
                begin = [0, 0, 0, 0]
                size_shape_0 = self._batch_size if mode == tf.estimator.ModeKeys.TRAIN else 1
                size = [size_shape_0, seq_len, seq_len, self.net.threshold]
                contact_pred_cm = tf.reduce_sum(tf.slice(
                    contact_pred, begin, size),
                    axis=3)
            else:
                contact_pred_cm = tf.math.exp(-1 * contact_pred)

            train_op = None
            if mode == tf.estimator.ModeKeys.TRAIN:
                train_op = tf.group(self._get_opt_op(loss, global_step),
                                    global_step)
            metrics = self._get_metrics_dict(contact_pred_cm, cm, seq_len,
                                             features)
            logging_hook = tf.train.LoggingTensorHook(
                {
                    "loss": loss,
                    'max': max,
                    'step': global_step
                },
                every_n_iter=100)

            return tf.estimator.EstimatorSpec(mode,
                                              loss=loss,
                                              train_op=train_op,
                                              eval_metric_ops=metrics,
                                              predictions=contact_pred,
                                              training_hooks=[logging_hook])

        return model_fn

    def train_input_fn(self):

        shapes, types = self.train_data_manager.get_required_shape_types()
        dataset = tf.data.Dataset.from_generator(
            self.train_data_manager.generator,
            output_types=types,
            output_shapes=shapes)
        dataset = dataset.batch(self._batch_size)
        return dataset

    def eval_input_fn(self):
        shapes, types = self.eval_data_manager.get_required_shape_types()
        dataset = tf.data.Dataset.from_generator(
            self.eval_data_manager.generator,
            output_types=types,
            output_shapes=shapes)
        dataset = dataset.batch(1)
        return dataset

    def predict_input_fn(self):
        shapes, types = self.predict_data_manager.get_required_shape_types()
        dataset = tf.data.Dataset.from_generator(
            self.predict_data_manager.generator,
            output_types=types,
            output_shapes=shapes)
        dataset = dataset.batch(1)
        return dataset

    def _get_train_spec(self):
        # Returns the train spec

        steps = self._train_params[
                    'epochs'] * self.train_data_manager.num_proteins

        train_spec = tf.estimator.TrainSpec(self.train_input_fn,
                                            max_steps=steps)
        return train_spec

    def _get_eval_spec(self):
        # Returns the eval spec
        eval_spec = tf.estimator.EvalSpec(
            self.eval_input_fn,
            steps=self.eval_data_manager.num_proteins,
            throttle_secs=1)
        return eval_spec

    def train_and_evaluate(self):
        # train and evaluate the estimator
        eval_spec = self._get_eval_spec()
        train_spec = self._get_train_spec()
        tf.estimator.train_and_evaluate(self.estimator, train_spec, eval_spec)

    def get_predictions_generator(self):
        return self.estimator.predict(self.predict_input_fn,
                                      yield_single_examples=True)


class DataGenerator:
    """Generator for model fn"""

    def __init__(self,
                 arch,
                 conv_features,
                 resnet_features,
                 mode,
                 epochs,
                 proteins,
                 k,
                 num_bins,
                 model_path,
                 log_pssm=True,
                 batch_size=1):
        self._arch = arch
        self._conv_features = conv_features
        self._resnet_features = resnet_features
        self._all_features = conv_features + resnet_features
        self._mode = mode
        self._epochs = epochs
        self._proteins = proteins
        self._log_pssm = log_pssm
        self._is_random = self._proteins is None
        self._k = k
        self._yielded = []
        self._all_features = conv_features + resnet_features
        self._include_local_dist = self._is_local_dist()
        self._num_bins = num_bins
        self._model_path = model_path
        self._batch_size = batch_size

    def _is_local_dist(self):

        if 'reference_dm_local_dist' in self._all_features:
            return True
        if 'k_reference_dm_local_dist' in self._all_features:
            return True
        if 'k_reference_dm_conv_local_dist' in self._all_features:
            return True
        return False

    @property
    def num_proteins(self):
        if self._proteins is None:
            return 20
        elif type(self._proteins) == list:
            return len(self._proteins)
        elif type(self._proteins) == dict:
            n_prots = 0
            for _, prots in self._proteins.items():
                n_prots += len(prots)
            return n_prots

    def get_required_shape_types(self):
        # shape and type for the dataset

        if self._arch == ARCHS.conv:
            shapes = {f: FEATURES_DIMS[f] for f in self._conv_features}
            types = {f: tf.float32 for f in self._conv_features}

            shapes['sequence_length'] = (1,)
            types['sequence_length'] = tf.int32

            if 'k_reference_dm' in shapes:
                shapes['k_reference_dm'] = (None, None, 1, self._k)
            if 'k_reference_dm_local_dist' in shapes:
                shapes['k_reference_dm_local_dist'] = (None, None, 2, self._k)
            if 'k_reference_dm_conv' in shapes:
                shapes['k_reference_dm_conv'] = (None, None, self._k)
            if 'k_reference_dm_conv_local_dist' in shapes:
                shapes['k_reference_dm_conv_local_dist'] = (None, None,
                                                            self._k * 2)
            if self._mode != 'predict':
                shapes['contact_map'] = (None, None, 1)
                types['contact_map'] = tf.float32

            return shapes, types

        elif self._arch == ARCHS.references_resnet:

            shapes = {f: FEATURES_DIMS[f] for f in self._all_features}
            types = {f: tf.float32 for f in self._all_features}
            if 'k_reference_dm' in shapes:
                shapes['k_reference_dm'] = (None, None, 1, self._k)
            if 'k_reference_dm_local_dist' in shapes:
                shapes['k_reference_dm_local_dist'] = (None, None, 2, self._k)
            shapes['sequence_length'] = (1,)
            types['sequence_length'] = tf.int32

            if self._mode != 'predict':
                shapes['contact_map'] = (None, None, 1)
                types['contact_map'] = tf.float32

            return shapes, types

        elif self._arch == ARCHS.multi_structure:
            shapes = {
                FEATURES.seq_refs: (None, PROTEIN_BOW_DIM, self._k),
                FEATURES.seq_target: (None, PROTEIN_BOW_DIM),
                FEATURES.plmc_score: (None, None, 1),
                FEATURES.k_reference_dm_conv: (None, None, self._k)
            }

            if self._mode != 'predict':
                shapes['contact_map'] = (None, None, 1)

            types = {f: tf.float32 for f in shapes}
            shapes['sequence_length'] = (1,)
            types['sequence_length'] = tf.int32

            return shapes, types

        elif self._arch == ARCHS.multi_structure_ccmpred or self._arch == ARCHS.multi_structure_ccmpred_2:
            shapes = {
                FEATURES.seq_refs: (None, PROTEIN_BOW_DIM, self._k),
                FEATURES.seq_target: (None, PROTEIN_BOW_DIM),
                FEATURES.plmc_score: (None, None, 1),
                FEATURES.k_reference_dm_conv: (None, None, self._k),
                FEATURES.ccmpred: (None, None, 1)
            }

            if self._mode != 'predict':
                shapes['contact_map'] = (None, None, 1)

            types = {f: tf.float32 for f in shapes}
            shapes['sequence_length'] = (1,)
            types['sequence_length'] = tf.int32

            return shapes, types

        elif self._arch == ARCHS.multi_structure_pssm:
            shapes = {
                FEATURES.seq_refs_pssm: (None, PROTEIN_BOW_DIM_PSSM, self._k),
                FEATURES.seq_target_pssm: (None, PROTEIN_BOW_DIM_PSSM),
                FEATURES.plmc_score: (None, None, 1),
                FEATURES.k_reference_dm_conv: (None, None, self._k)
            }

            if self._mode != 'predict':
                shapes['contact_map'] = (None, None, 1)

            types = {f: tf.float32 for f in shapes}
            shapes['sequence_length'] = (1,)
            types['sequence_length'] = tf.int32

            return shapes, types

        elif self._arch == ARCHS.ms_ccmpred_pssm:
            shapes = {
                FEATURES.seq_refs_pssm: (None, PROTEIN_BOW_DIM_PSSM, self._k),
                FEATURES.seq_target_pssm: (None, PROTEIN_BOW_DIM_PSSM),
                FEATURES.plmc_score: (None, None, 1),
                FEATURES.k_reference_dm_conv: (None, None, self._k),
                FEATURES.ccmpred: (None, None, 1)
            }

            if self._mode != 'predict':
                shapes['contact_map'] = (None, None, 1)

            types = {f: tf.float32 for f in shapes}
            shapes['sequence_length'] = (1,)
            types['sequence_length'] = tf.int32

            return shapes, types
        elif self._arch == ARCHS.ms_ss_ccmpred_pssm:
            shapes = {
                FEATURES.seq_refs_pssm_ss: (None, PROTEIN_BOW_DIM_PSSM_SS, self._k),
                FEATURES.seq_target_pssm_ss: (None, PROTEIN_BOW_DIM_PSSM_SS),
                FEATURES.plmc_score: (None, None, 1),
                FEATURES.k_reference_dm_conv: (None, None, self._k),
                FEATURES.ccmpred: (None, None, 1)
            }

            if self._mode != 'predict':
                shapes['contact_map'] = (None, None, 1)

            types = {f: tf.float32 for f in shapes}
            shapes['sequence_length'] = (1,)
            types['sequence_length'] = tf.int32

            return shapes, types
        elif self._arch == ARCHS.ms_ss_ccmpred or self._arch ==  ARCHS.ms_ss_ccmpred_2:
            shapes = {
                FEATURES.seq_refs_ss: (None, PROTEIN_BOW_DIM_SS, self._k),
                FEATURES.seq_target_ss: (None, PROTEIN_BOW_DIM_SS),
                FEATURES.plmc_score: (None, None, 1),
                FEATURES.k_reference_dm_conv: (None, None, self._k),
                FEATURES.ccmpred: (None, None, 1)
            }

            if self._mode != 'predict':
                shapes['contact_map'] = (None, None, 1)

            types = {f: tf.float32 for f in shapes}
            shapes['sequence_length'] = (1,)
            types['sequence_length'] = tf.int32

            return shapes, types

    def _yield_random_data(self, l):
        data = {}
        if self._mode != 'predict':
            data['contact_map'] = np.array(
                np.random.randint(low=0, high=2, size=(l, l, 1)), np.float32)
        if self._arch == ARCHS.multi_structure:
            data[FEATURES.seq_refs] = np.random.random(
                (l, PROTEIN_BOW_DIM, self._k))
            data[FEATURES.seq_target] = np.random.random((l, PROTEIN_BOW_DIM))
            data[FEATURES.plmc_score] = np.random.random((l, l, 1))
            data[FEATURES.k_reference_dm_conv] = np.random.random(
                (l, l, self._k))
            data['sequence_length'] = np.array([l], dtype=np.int32)

            return data

        if self._arch == ARCHS.multi_structure_ccmpred or self._arch == ARCHS.multi_structure_ccmpred_2:
            data[FEATURES.seq_refs] = np.random.random(
                (l, PROTEIN_BOW_DIM, self._k))
            data[FEATURES.seq_target] = np.random.random((l, PROTEIN_BOW_DIM))
            data[FEATURES.plmc_score] = np.random.random((l, l, 1))
            data[FEATURES.ccmpred] = np.random.random((l, l, 1))
            data[FEATURES.k_reference_dm_conv] = np.random.random(
                (l, l, self._k))
            data['sequence_length'] = np.array([l], dtype=np.int32)

            return data

        if self._arch == ARCHS.multi_structure_pssm:
            data[FEATURES.seq_refs_pssm] = np.random.random(
                (l, PROTEIN_BOW_DIM_PSSM, self._k))
            data[FEATURES.seq_target_pssm] = np.random.random((l, PROTEIN_BOW_DIM_PSSM))
            data[FEATURES.plmc_score] = np.random.random((l, l, 1))
            data[FEATURES.k_reference_dm_conv] = np.random.random(
                (l, l, self._k))
            data['sequence_length'] = np.array([l], dtype=np.int32)

            return data
        if self._arch == ARCHS.ms_ccmpred_pssm:
            data[FEATURES.seq_refs_pssm] = np.random.random(
                (l, PROTEIN_BOW_DIM_PSSM, self._k))
            data[FEATURES.seq_target_pssm] = np.random.random((l, PROTEIN_BOW_DIM_PSSM))
            data[FEATURES.plmc_score] = np.random.random((l, l, 1))
            data[FEATURES.ccmpred] = np.random.random((l, l, 1))
            data[FEATURES.k_reference_dm_conv] = np.random.random(
                (l, l, self._k))
            data['sequence_length'] = np.array([l], dtype=np.int32)

            return data
        if self._arch == ARCHS.ms_ss_ccmpred_pssm:
            data[FEATURES.seq_refs_pssm_ss] = np.random.random(
                (l, PROTEIN_BOW_DIM_PSSM_SS, self._k))
            data[FEATURES.seq_target_pssm_ss] = np.random.random((l, PROTEIN_BOW_DIM_PSSM_SS))
            data[FEATURES.plmc_score] = np.random.random((l, l, 1))
            data[FEATURES.ccmpred] = np.random.random((l, l, 1))
            data[FEATURES.k_reference_dm_conv] = np.random.random(
                (l, l, self._k))
            data['sequence_length'] = np.array([l], dtype=np.int32)

            return data

        if self._arch == ARCHS.ms_ss_ccmpred or self._arch == ARCHS.ms_ss_ccmpred_2:
            data[FEATURES.seq_refs_ss] = np.random.random(
                (l, PROTEIN_BOW_DIM_SS, self._k))
            data[FEATURES.seq_target_ss] = np.random.random((l, PROTEIN_BOW_DIM_SS))
            data[FEATURES.plmc_score] = np.random.random((l, l, 1))
            data[FEATURES.ccmpred] = np.random.random((l, l, 1))
            data[FEATURES.k_reference_dm_conv] = np.random.random(
                (l, l, self._k))
            data['sequence_length'] = np.array([l], dtype=np.int32)

            return data
        FEATURES_DIMS['k_reference_dm_conv'] = (None, None, self._k)
        FEATURES_DIMS['k_reference_dm_conv_local_dist'] = (None, None,
                                                           self._k * 2)

        data.update({
            f: np.array(np.random.random([l, l, FEATURES_DIMS[f][2]]),
                        np.float32)
            for f in self._conv_features
        })
        data['sequence_length'] = np.array([l], dtype=np.int32)

        if self._arch == ARCHS.conv:
            return data

        elif self._arch == ARCHS.references_resnet:

            data['k_reference_dm'] = np.random.random((l, l, 1, self._k))

        return data

    def _bin_array(self, data_handler):
        """Returns the correct label given the number of bins

        Args:
            data_handler (ProteinDataHandler): out data handler

        Returns:
            np.array: binned label of shape (l, l, num_bins)

        """
        if self._num_bins == 1:
            return np.expand_dims(data_handler.target_pdb_dm, axis=2)

        if self._num_bins == 2:
            return np.expand_dims(data_handler.target_pdb_cm, axis=2)

        distance_matrix = data_handler.target_pdb_dm

        bin_step = 24 / self._num_bins

        bins = [-1.1] + [
            np.round(bin_step * i, 2) for i in range(self._num_bins - 1)
        ] + [100]
        bins_shifted = [0] + [
            np.round(bin_step * (i + 1), 2) for i in range(self._num_bins - 1)
        ] + [100]

        binned_distance_matrix = np.array(np.logical_and(
            distance_matrix[..., None] >= np.array(bins),
            distance_matrix[..., None] <= np.array(bins_shifted)),
            dtype=np.float32)

        return binned_distance_matrix

    def _pad_feature(self, l, feature):

        if len(feature.shape) == 3:
            padding = np.zeros((l, l, feature.shape[2]))
        elif len(feature.shape) == 4:
            padding = np.zeros((l, l, feature.shape[2], feature.shape[3]))

        feature_dim = feature.shape[0]

        padding[0:feature_dim, 0:feature_dim, ...] = feature

        return padding

    def _yield_protein_data(self, protein, l=None):

        data = {}
        data_handler = ProteinDataHandler(protein,
                                          k=self._k,
                                          include_local_dist=False,
                                          structures_version=3,
                                          log_pssm=self._log_pssm)

        if self._arch == ARCHS.multi_structure:

            try:
                data[FEATURES.plmc_score] = np.expand_dims(
                    data_handler.plmc_score, axis=2)
                data[FEATURES.
                    k_reference_dm_conv] = data_handler.k_reference_dm_conv
                data[FEATURES.seq_refs] = data_handler.seq_refs
                data[FEATURES.seq_target] = data_handler.seq_target
            except AttributeError:
                return
            has_nones = False
            for f in data:
                has_nones |= data[f] is None
            if has_nones:
                return

        elif self._arch == ARCHS.multi_structure_ccmpred or self._arch == ARCHS.multi_structure_ccmpred_2:

            try:
                data[FEATURES.plmc_score] = np.expand_dims(
                    data_handler.plmc_score, axis=2)
                data[FEATURES.
                    k_reference_dm_conv] = data_handler.k_reference_dm_conv
                data[FEATURES.seq_refs] = data_handler.seq_refs
                data[FEATURES.seq_target] = data_handler.seq_target
                data[FEATURES.ccmpred] = np.expand_dims(data_handler.ccmpred, axis=2)

            except (AttributeError, np.AxisError):
                return
            has_nones = False
            for f in data:
                has_nones |= data[f] is None
            if has_nones:
                return
        elif self._arch == ARCHS.ms_ccmpred_pssm:

            try:
                data[FEATURES.plmc_score] = np.expand_dims(
                    data_handler.plmc_score, axis=2)
                data[FEATURES.
                    k_reference_dm_conv] = data_handler.k_reference_dm_conv
                data[FEATURES.seq_refs_pssm] = data_handler.seq_refs_pssm
                data[FEATURES.seq_target_pssm] = data_handler.seq_target_pssm
                data[FEATURES.ccmpred] = np.expand_dims(data_handler.ccmpred, axis=2)

            except (AttributeError, np.AxisError):
                return
            has_nones = False
            for f in data:
                has_nones |= data[f] is None
            if has_nones:
                return
        elif self._arch == ARCHS.ms_ss_ccmpred_pssm:

            try:
                data[FEATURES.plmc_score] = np.expand_dims(
                    data_handler.plmc_score, axis=2)
                data[FEATURES.
                    k_reference_dm_conv] = data_handler.k_reference_dm_conv
                data[FEATURES.seq_refs_pssm_ss] = data_handler.seq_refs_pssm_ss
                data[FEATURES.seq_target_pssm_ss] = data_handler.seq_target_pssm_ss
                data[FEATURES.ccmpred] = np.expand_dims(data_handler.ccmpred, axis=2)

            except (AttributeError, np.AxisError):
                return
            has_nones = False
            for f in data:
                has_nones |= data[f] is None
            if has_nones:
                return
        elif self._arch == ARCHS.ms_ss_ccmpred or self._arch == ARCHS.ms_ss_ccmpred_2:

            try:
                data[FEATURES.plmc_score] = np.expand_dims(
                    data_handler.plmc_score, axis=2)
                data[FEATURES.
                    k_reference_dm_conv] = data_handler.k_reference_dm_conv
                data[FEATURES.seq_refs_ss] = data_handler.seq_refs_ss
                data[FEATURES.seq_target_ss] = data_handler.seq_target_ss
                data[FEATURES.ccmpred] = np.expand_dims(data_handler.ccmpred, axis=2)

            except (AttributeError, np.AxisError):
                return
            has_nones = False
            for f in data:
                has_nones |= data[f] is None

            if has_nones:
                return

        elif self._arch == ARCHS.multi_structure_pssm:

            try:
                data[FEATURES.plmc_score] = np.expand_dims(
                    data_handler.plmc_score, axis=2)
                data[FEATURES.
                    k_reference_dm_conv] = data_handler.k_reference_dm_conv
                data[FEATURES.seq_refs_pssm] = data_handler.seq_refs_pssm
                data[FEATURES.seq_target_pssm] = data_handler.seq_target_pssm
            except AttributeError:
                return
            has_nones = False
            for f in data:
                has_nones |= data[f] is None
            if has_nones:
                return

        else:

            for f in self._all_features:

                if not hasattr(data_handler, f):
                    return

                feature = getattr(data_handler, f)
                if feature is None:
                    return
                if len(feature.shape) == 2:
                    feature = np.expand_dims(feature, axis=2)
                data[f] = feature if l is None else self._pad_feature(
                    l, feature)

        data['sequence_length'] = np.array(
            [data_handler.target_sequence_length])

        if data_handler.target_sequence_length > 1200:
            return

        if self._mode != 'predict':
            cm = self._bin_array(data_handler)
            data['contact_map'] = cm if l is None else self._pad_feature(l, cm)

        if self._mode == 'predict':
            self._yielded.append(protein)

        return data

    def generator(self):
        if self._is_random:
            LOGGER.info('using random data')
            for epoch in range(self._epochs):
                for i in range(self.num_proteins):
                    l = np.random.choice([7, 8, 9])

                    for b in range(self._batch_size):
                        yield self._yield_random_data(l)
        elif self._batch_size == 1:
            yielded = []
            prots = self._proteins
            for epoch in range(self._epochs):
                random.shuffle(prots)
                for protein in prots:
                    LOGGER.info(f'Protein {protein}')
                    data = self._yield_protein_data(protein)
                    if data is not None:
                        if self._mode == 'predict':
                            yielded.append(protein)
                        yield data
                    else:
                        LOGGER.info(f'No data yielded for {protein}')

            if self._mode == 'predict':
                pkl_save(filename=os.path.join(self._model_path,
                                               'predicted_proteins.pkl'),
                         data=yielded)
        else:
            n_proteins = 0
            for grp, prots in self._proteins.items():
                n_proteins += len(prots)
            LOGGER.info('Number of proteins is %s' % n_proteins)

            total_steps = self._epochs * n_proteins
            batches = np.ceil(total_steps / self._batch_size)
            LOGGER.info('Number of batches %s' % batches)

            lengths = list(self._proteins.keys())
            batched_delivered = 0

            while batched_delivered < batches:
                yielded = 0
                l = np.random.choice(lengths)

                if len(self._proteins[l]) < self._batch_size:
                    continue

                while yielded < self._batch_size:
                    protein = np.random.choice(self._proteins[l])
                    data = self._yield_protein_data(protein, l)

                    if data is not None:
                        yielded += 1
                        yield data

                batched_delivered += 1
