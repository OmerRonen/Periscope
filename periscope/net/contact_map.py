import logging
import os
import warnings

import tensorflow as tf
import numpy as np

from functools import partial

from .params import NetParams
from ..data.generator import Generators
from ..utils.constants import (ARCHS, PROTEIN_BOW_DIM, PROTEIN_BOW_DIM_PSSM_SS,
                               PROTEIN_BOW_DIM_SS,
                               DATASETS, FEATURES, PATHS)
from ..net.basic_ops import (get_top_category_accuracy, deep_conv_op, multi_structures_op,
                             multi_structures_op_simple, get_opt_op, upper_triangular_mse_loss,
                             upper_triangular_cross_entropy_loss, periscope_op, evo_op, template_op)
from ..utils.utils import yaml_save, pkl_load, check_path, get_quant, pkl_save

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
warnings.filterwarnings('ignore', module='/tensorflow/')


class ProteinNet:
    """neural net that outputs predicted contact map"""

    def __init__(self, net_params):
        BIN_THRESHOLDS = {1: 8, 2: 1}

        self.arch = net_params['architecture']['arch']

        self._conv_arch_params = net_params['architecture']['conv']
        ms = net_params['architecture'].get('ms', {"deep_projection": False})
        self._deep_projection = ms['deep_projection']

        self._data_params = net_params['data']
        self._features = self._data_params['conv_features']
        self.num_bins = self._conv_arch_params['num_bins']
        self._dilation = self._conv_arch_params['dilation']
        self.output_shape = (None, None, self.num_bins)

        self.prediction_type = 'regression' if self.num_bins == 1 else 'classification'
        self.threshold = 8 if self.num_bins == 1 else int(self.num_bins / 2)

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
        raise NotImplementedError

    def get_inputs(self):
        raise NotImplementedError


class PeriscopeNet(ProteinNet):
    @property
    def predicted_contact_map(self):
        return partial(periscope_op,
                       conv_params=self._conv_arch_params)

    # @property
    # def weights(self):
    #     return partial(get_weights,
    #                    conv_params=self._conv_arch_params)

    def get_inputs(self, features):
        periscope_input = {
            'dms': features[FEATURES.k_reference_dm_conv],
            'seq_refs': features[FEATURES.seq_refs],
            # 'evfold': features[FEATURES.evfold],
            'ccmpred': features[FEATURES.ccmpred],
            FEATURES.pwm_evo: features[FEATURES.pwm_evo],
            FEATURES.pwm_w: features[FEATURES.pwm_w],
            FEATURES.conservation: features[FEATURES.conservation],
            FEATURES.beff: features[FEATURES.beff]
        }
        return periscope_input


class TempaltesNet(ProteinNet):
    @property
    def predicted_contact_map(self):
        return partial(template_op,
                       conv_params=self._conv_arch_params)

    # @property
    # def weights(self):
    #     return partial(get_weights,
    #                    conv_params=self._conv_arch_params)

    def get_inputs(self, features):
        periscope_input = {
            'dms': features[FEATURES.k_reference_dm_conv],
            'seq_refs': features[FEATURES.seq_refs],
            FEATURES.pwm_w: features[FEATURES.pwm_w],
        }
        return periscope_input


class EvoNet(ProteinNet):
    @property
    def predicted_contact_map(self):
        return partial(evo_op,
                       conv_params=self._conv_arch_params)

    # @property
    # def weights(self):
    #     return partial(get_weights,
    #                    conv_params=self._conv_arch_params)

    def get_inputs(self, features):
        periscope_input = {
            'evfold': features[FEATURES.evfold],
            'ccmpred': features[FEATURES.ccmpred],
            FEATURES.pwm_evo: features[FEATURES.pwm_evo],
            FEATURES.conservation: features[FEATURES.conservation],
            FEATURES.beff: features[FEATURES.beff]
        }
        return periscope_input


class ConvProteinNet(ProteinNet):
    @property
    def predicted_contact_map(self):
        return partial(deep_conv_op, **self._conv_arch_params)

    def get_inputs(self, features):
        features_dict = {
            f: d
            for f, d in features.items() if f in self._features
        }

        inputs = {
            'conv_input_tensor': tf.concat(list(features_dict.values()),
                                           axis=3)
        }

        return inputs


class MsCCmpredProteinNet(ProteinNet):
    @property
    def predicted_contact_map(self):
        return partial(multi_structures_op,
                       conv_params=self._conv_arch_params,
                       deep_projection=self._deep_projection)

    def get_inputs(self, features):
        multi_structures_input = {
            'dms': features[FEATURES.k_reference_dm_conv],
            'seq_refs': features[FEATURES.seq_refs],
            'seq_target': features[FEATURES.seq_target],
            'evfold': features[FEATURES.evfold],
            'ccmpred': features[FEATURES.ccmpred]
        }
        return multi_structures_input


class MsCCmpredProteinNetSimple(ProteinNet):
    @property
    def predicted_contact_map(self):
        return partial(multi_structures_op_simple,
                       conv_params=self._conv_arch_params,
                       deep_projection=self._deep_projection)

    def get_inputs(self, features):
        multi_structures_input = {
            'dms': features[FEATURES.k_reference_dm_conv],
            'seq_refs': features[FEATURES.seq_refs],
            'seq_target': features[FEATURES.seq_target],
            'evfold': features[FEATURES.evfold],
            'ccmpred': features[FEATURES.ccmpred]
        }
        return multi_structures_input


class MsSsCCmpredProteinNet(ProteinNet):
    @property
    def predicted_contact_map(self):
        return partial(multi_structures_op_simple,
                       conv_params=self._conv_arch_params,
                       prot_dim=PROTEIN_BOW_DIM_SS,
                       k=100)

    def get_inputs(self, features):
        multi_structures_input = {
            'dms': features[FEATURES.k_reference_dm_conv],
            'seq_refs': features[FEATURES.seq_refs_ss],
            'seq_target': features[FEATURES.seq_target_ss],
            'evfold': features[FEATURES.evfold],
            'ccmpred': features[FEATURES.ccmpred]
        }
        return multi_structures_input


class MsSsCCmpredPssmProteinNet(ProteinNet):
    @property
    def predicted_contact_map(self):
        return partial(multi_structures_op_simple,
                       conv_params=self._conv_arch_params,
                       prot_dim=PROTEIN_BOW_DIM_PSSM_SS,
                       k=100)

    def get_inputs(self, features):
        multi_structures_input = {
            'dms': features[FEATURES.k_reference_dm_conv],
            'seq_refs': features[FEATURES.seq_refs_pssm_ss],
            'seq_target': features[FEATURES.seq_target_pssm_ss],
            'evfold': features[FEATURES.evfold],
            'ccmpred': features[FEATURES.ccmpred]
        }
        return multi_structures_input


_nets = {ARCHS.conv: ConvProteinNet,
         ARCHS.multi_structure_ccmpred: MsCCmpredProteinNet,
         ARCHS.multi_structure_ccmpred_2: MsCCmpredProteinNetSimple,
         ARCHS.ms_ss_ccmpred: MsSsCCmpredProteinNet,
         ARCHS.ms_ss_ccmpred_pssm: MsSsCCmpredPssmProteinNet,
         ARCHS.periscope: PeriscopeNet,
         ARCHS.periscope2: PeriscopeNet,
         ARCHS.templates: TempaltesNet,
         ARCHS.evo: EvoNet}


def get_model_by_name(model_name, test_dataset=None):
    params = NetParams(
        os.path.join(PATHS.models, model_name,
                     'params.yml')).params
    params['train']['path'] = os.path.join(PATHS.models, model_name)
    if test_dataset is not None:
        params['train']['test_dataset'] = test_dataset

    return ContactMapEstimator(params)


class ContactMapEstimator:
    """Contact map predictor"""

    def __init__(self, params, old=False):
        self._train_params = params['train']
        self.path = self._train_params['path']
        self.name = self._train_params['path'].split('/')[-1]

        self._old = old

        check_create_folder(self.path)
        self.artifacts_path = os.path.join(self.path, 'artifacts')
        check_create_folder(self.artifacts_path)

        params_file = os.path.join(self.path, 'params.yml')
        if not os.path.isfile(params_file):
            yaml_save(filename=params_file, data=params)

        self._opt_params = self._train_params['opt']

        self._batch_size = self._train_params['batch_size']
        self._require_template = self._train_params['require_template']

        self.conv_features = params['net']['data']['conv_features']

        self._arch = params['net']['architecture']['arch']
        self.net = _nets[self._arch](params['net'])

        k = params['net']['data']['k']

        train_dataset_name = self._train_params['train_dataset']
        eval_dataset_name = self._train_params['eval_dataset']
        test_dataset_name = self._train_params['test_dataset']

        train_proteins = None if train_dataset_name is None else getattr(DATASETS, train_dataset_name)
        eval_proteins = None if eval_dataset_name is None else getattr(DATASETS, eval_dataset_name)
        test_proteins = None if test_dataset_name is None else getattr(DATASETS, test_dataset_name)

        self._data_generator_args = {
            'conv_features': self.conv_features,
            'n_refs': k,
            'model_path': self.artifacts_path,
            'num_bins': self.net.num_bins,
            'require_template': self._require_template
        }
        train_epochs = self._train_params['epochs']
        t_drop = self._train_params['templates_dropout']

        eval_epochs = 1
        test_epochs = 1
        self._generator = Generators[self._arch]
        self.train_data_manager = self._generator(proteins=train_proteins,
                                                  mode=tf.estimator.ModeKeys.TRAIN,
                                                  epochs=train_epochs,
                                                  batch_size=self._batch_size,
                                                  dataset=train_dataset_name,
                                                  templates_dropout=t_drop,
                                                  **self._data_generator_args)
        self.eval_data_manager = self._generator(proteins=eval_proteins,
                                                 mode=tf.estimator.ModeKeys.TRAIN,
                                                 epochs=eval_epochs,
                                                 dataset=eval_dataset_name,
                                                 **self._data_generator_args)
        self.predict_data_manager = self._generator(proteins=test_proteins,
                                                    mode=tf.estimator.ModeKeys.PREDICT,
                                                    epochs=test_epochs,
                                                    dataset=test_dataset_name,
                                                    old=self._old,
                                                    **self._data_generator_args)

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
                                                params={"L": 5},
                                                config=config)

    def predict(self, proteins=None, dataset=None, family=None):

        dataset = self.predict_data_manager.dataset if dataset is None else dataset
        predictions_path = os.path.join(self.path, 'predictions')
        check_path(predictions_path)
        prediction_file = os.path.join(predictions_path, f'{dataset}.pkl')
        if os.path.isfile(prediction_file) and proteins is None:
            return pkl_load(prediction_file)
        if proteins is not None:
            LOGGER.info(f'require template {self._require_template}')

            preds = list(self.get_custom_predictions_gen(proteins, dataset, family,
                                                         require_template=self._require_template))
        else:
            preds = list(self.get_predictions_generator())

        proteins = pkl_load(
            os.path.join(self.artifacts_path, f'predicted_proteins_{dataset}.pkl'))
        model_predictions = {'cm': {}, 'logits': {}, 'weights': {}}

        for protein, pred in dict(zip(proteins, preds)).items():
            logits = pred['cm']
            weights = pred['weights']
            shp = int(logits.shape[-1])
            contact_probs = np.sum(logits[..., 0:int(shp / 2)], axis=-1)
            l = contact_probs.shape[1]
            # top l/2 predictions
            quant = get_quant(l)
            model_predictions['cm'][protein] = np.where(
                contact_probs >= np.quantile(
                    contact_probs[np.triu_indices(contact_probs.shape[0])], quant),
                1, 0)

            model_predictions['logits'][protein] = contact_probs
            model_predictions['weights'][protein] = weights

        if proteins is None:
            pkl_save(filename=prediction_file, data=model_predictions)

        return model_predictions

    def _get_opt_op(self, loss, global_step):
        return get_opt_op(loss, global_step, **self._opt_params)

    def _get_metrics_dict(self, contact_pred_cm, cm, seq_len, features_dict):
        metrics_dict = self.net.get_top_prediction(contact_pred_cm, cm,
                                                   seq_len, 'eval')
        # metrics_dict.update(
        #     self.net.define_prediction_methods_similarity(
        #         contact_pred_cm, features_dict, seq_len, 'eval'))

        return metrics_dict

    def _get_model_fn(self):
        def model_fn(features, labels, mode, params):

            inputs = self.net.get_inputs(features)
            seq_len = features['sequence_length'][0, 0]

            contact_pred, weights = self.net.predicted_contact_map(**inputs)

            max_val = tf.reduce_max(contact_pred)

            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(mode,
                                                  predictions={'cm': contact_pred, 'weights': weights})

            cm = features['contact_map']

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
                cm = tf.reduce_sum(tf.slice(
                    cm, begin, size),
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
                    'max': max_val,
                    'step': global_step
                },
                every_n_iter=100)
            spec = tf.estimator.EstimatorSpec(mode,
                                              loss=loss,
                                              train_op=train_op,
                                              eval_metric_ops=metrics,
                                              predictions=contact_pred,
                                              training_hooks=[logging_hook])

            return spec

        return model_fn

    def train_input_fn(self):

        shapes, types = self.train_data_manager.required_shape_types
        dataset = tf.data.Dataset.from_generator(
            self.train_data_manager.generator,
            output_types=types,
            output_shapes=shapes)
        dataset = dataset.batch(self._batch_size)
        return dataset

    def eval_input_fn(self):
        shapes, types = self.eval_data_manager.required_shape_types
        dataset = tf.data.Dataset.from_generator(
            self.eval_data_manager.generator,
            output_types=types,
            output_shapes=shapes)
        dataset = dataset.batch(1)
        return dataset

    def predict_input_fn(self):
        shapes, types = self.predict_data_manager.required_shape_types
        dataset = tf.data.Dataset.from_generator(
            self.predict_data_manager.generator,
            output_types=types,
            output_shapes=shapes)
        dataset = dataset.batch(1)
        return dataset

    def _get_custom_input_fn(self, proteins, dataset, family=None, require_template=True):
        LOGGER.info(f'require template {require_template}')

        data_generator_args = self._data_generator_args
        data_generator_args['require_template'] = require_template
        LOGGER.info(data_generator_args)
        data_gen = self._generator(proteins=proteins,
                                   mode=tf.estimator.ModeKeys.PREDICT,
                                   epochs=1,
                                   dataset=dataset,
                                   family=family,
                                   **data_generator_args)

        def custom_input_fn():
            shapes, types = data_gen.required_shape_types
            dataset = tf.data.Dataset.from_generator(
                data_gen.generator,
                output_types=types,
                output_shapes=shapes)
            dataset = dataset.batch(1)
            return dataset

        return custom_input_fn

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

    def get_eval_predictions_generator(self):
        return self.estimator.predict(self.eval_input_fn,
                                      yield_single_examples=True)

    def get_custom_predictions_gen(self, proteins, dataset, family=None, require_template=True):
        LOGGER.info(f'require template {require_template}')

        input_fn = self._get_custom_input_fn(proteins, dataset, family, require_template=require_template)
        return self.estimator.predict(input_fn,
                                      yield_single_examples=True)

    @property
    def n_params(self):
        vars = self.estimator.get_variable_names()
        n_params = int(sum([np.prod(self.estimator.get_variable_value(var).shape) for var in vars]))
        return n_params
