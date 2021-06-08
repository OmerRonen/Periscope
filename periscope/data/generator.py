import logging
import os
import random
import tempfile
import time

import numpy as np
import tensorflow as tf

from .creator import DataCreator
from .seeker import DataSeeker
from ..utils.constants import (FEATURES, PROTEIN_BOW_DIM_PSSM_SS, PROTEIN_BOW_DIM_SS, PROTEIN_BOW_DIM,
                               FEATURES_DIMS, ARCHS)
from ..utils.utils import pkl_save, pkl_load, bin_array

logging.basicConfig(level=logging.INFO)

LOGGER = logging.getLogger(__name__)


class DataGenerator:
    def __init__(self, proteins, epochs, mode, model_path, dataset,
                 batch_size=1, conv_features=None, n_refs=None, num_bins=2, old=False,
                 templates_dropout=0, family=None, require_template=True):
        self._proteins = proteins
        self._is_random = self._proteins is None
        self._batch_size = batch_size
        self._epochs = epochs
        self._mode = mode
        self._conv_features = conv_features
        self._n_refs = n_refs
        self._model_path = model_path
        self._tmp_dir = tempfile.TemporaryDirectory(dir=self._model_path)
        LOGGER.info(f'Temp directory created at {self._tmp_dir.name}')
        self._yielded = []
        self._num_bins = num_bins
        self.dataset = dataset
        self._old = old
        self._templates_dropout = templates_dropout
        self._family = family
        self._require_template = require_template
        LOGGER.info(f'require_template generator {self._require_template}')

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

    @property
    def required_shape_types(self):
        raise NotImplemented

    def _yield_random_data(self, l):
        raise NotImplementedError

    def _yield_protein_data(self, protein, l=None):
        raise NotImplementedError

    def generator(self):
        if self._is_random:
            LOGGER.info('using random data')
            for epoch in range(self._epochs):
                for i in range(self.num_proteins):
                    l = 5  # np.random.choice([7, 8, 9])

                    for b in range(self._batch_size):
                        yield self._yield_random_data(l)
        elif self._batch_size == 1:
            yielded = []
            prots = self._proteins
            for epoch in range(self._epochs):
                random.shuffle(prots)
                for protein in prots:
                    data = self._yield_protein_data(protein)
                    if data is not None:
                        if self._mode == tf.estimator.ModeKeys.PREDICT:
                            yielded.append(protein)
                        yield data

            if self._mode == tf.estimator.ModeKeys.PREDICT:
                pkl_save(filename=os.path.join(self._model_path,
                                               f'predicted_proteins_{self.dataset}.pkl'),
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

    def _pad_feature(self, l, feature):

        if len(feature.shape) == 3:
            padding = np.zeros((l, l, feature.shape[2]))
        elif len(feature.shape) == 4:
            padding = np.zeros((l, l, feature.shape[2], feature.shape[3]))

        feature_dim = feature.shape[0]

        padding[0:feature_dim, 0:feature_dim, ...] = feature

        return padding


class MsSsCCmpredGenerator(DataGenerator):
    @property
    def required_shape_types(self):
        shapes = {
            FEATURES.seq_refs_ss: (None, PROTEIN_BOW_DIM_SS, self._n_refs),
            FEATURES.seq_target_ss: (None, PROTEIN_BOW_DIM_SS),
            FEATURES.evfold: (None, None, 1),
            FEATURES.k_reference_dm_conv: (None, None, self._n_refs),
            FEATURES.ccmpred: (None, None, 1)
        }

        if self._mode != tf.estimator.ModeKeys.PREDICT:
            shapes['contact_map'] = (None, None, 1)

        types = {f: tf.float32 for f in shapes}
        shapes['sequence_length'] = (1,)
        types['sequence_length'] = tf.int32

        return shapes, types

    def _yield_random_data(self, l):
        data = {}
        if self._mode != tf.estimator.ModeKeys.PREDICT:
            data['contact_map'] = np.array(
                np.random.randint(low=0, high=2, size=(l, l, 1)), np.float32)
        data[FEATURES.seq_refs_ss] = np.random.random(
            (l, PROTEIN_BOW_DIM_SS, self._n_refs))
        data[FEATURES.seq_target_ss] = np.random.random((l, PROTEIN_BOW_DIM_SS))
        data[FEATURES.evfold] = np.random.random((l, l, 1))
        data[FEATURES.ccmpred] = np.random.random((l, l, 1))
        data[FEATURES.k_reference_dm_conv] = np.random.random(
            (l, l, self._n_refs))
        data['sequence_length'] = np.array([l], dtype=np.int32)

        return data

    def _yield_protein_data(self, protein, l=None):
        data = {}

        data_seeker = DataSeeker(protein, n_refs=self._n_refs)
        data_creator = DataCreator(protein, self._n_refs)
        try:
            data[FEATURES.evfold] = np.expand_dims(
                data_seeker.evfold, axis=2)
            data[FEATURES.k_reference_dm_conv] = data_seeker.k_reference_dm_conv
            data[FEATURES.seq_refs_ss] = data_creator.seq_refs_ss
            data[FEATURES.seq_target_ss] = data_creator.seq_target_ss
            data[FEATURES.ccmpred] = np.expand_dims(data_seeker.ccmpred, axis=2)

        except (AttributeError, np.AxisError):
            return
        has_nones = False
        for f in data:
            has_nones |= data[f] is None

        if has_nones:
            return

        target_sequence_length = len(data_seeker.protein.sequence)
        data['sequence_length'] = np.array([target_sequence_length])

        if target_sequence_length > 1200:
            return

        if self._mode != tf.estimator.ModeKeys.PREDICT:
            cm = bin_array(data_seeker.protein.dm, self._num_bins)
            data['contact_map'] = cm if l is None else self._pad_feature(l, cm)

        if self._mode == tf.estimator.ModeKeys.PREDICT:
            self._yielded.append(protein)

        return data


class MsCCmpredGenerator(DataGenerator):
    @property
    def required_shape_types(self):
        shapes = {
            FEATURES.seq_refs: (None, PROTEIN_BOW_DIM, self._n_refs),
            FEATURES.seq_target: (None, PROTEIN_BOW_DIM),
            FEATURES.evfold: (None, None, 1),
            FEATURES.k_reference_dm_conv: (None, None, self._n_refs),
            FEATURES.ccmpred: (None, None, 1)
        }

        if self._mode != tf.estimator.ModeKeys.PREDICT:
            shapes['contact_map'] = (None, None, 1)

        types = {f: tf.float32 for f in shapes}
        shapes['sequence_length'] = (1,)
        types['sequence_length'] = tf.int32

        return shapes, types

    def _yield_random_data(self, l):
        data = {}
        if self._mode != tf.estimator.ModeKeys.PREDICT:
            data['contact_map'] = np.array(
                np.random.randint(low=0, high=2, size=(l, l, 1)), np.float32)
        data[FEATURES.seq_refs] = np.random.random(
            (l, PROTEIN_BOW_DIM, self._n_refs))
        data[FEATURES.seq_target] = np.random.random((l, PROTEIN_BOW_DIM))
        data[FEATURES.evfold] = np.random.random((l, l, 1))
        data[FEATURES.ccmpred] = np.random.random((l, l, 1))
        data[FEATURES.k_reference_dm_conv] = np.random.random(
            (l, l, self._n_refs))
        data['sequence_length'] = np.array([l], dtype=np.int32)

        return data

    def _yield_protein_data(self, protein, l=None):

        file = os.path.join(self._tmp_dir.name, f'{protein}.pkl')
        if os.path.isfile(file):
            LOGGER.info(f"loading {file}")
            return pkl_load(file)

        data = {}
        data_seeker = DataSeeker(protein, n_refs=self._n_refs)
        data_creator = DataCreator(protein, n_refs=self._n_refs)
        LOGGER.info(protein)

        try:
            # start_time = time.time()
            evfold = np.expand_dims(data_seeker.evfold, axis=2)
            # end_time = time.time()
            # LOGGER.info(f'Evfold takes  {end_time-start_time}')
            # start_time = time.time()
            ccmpred = np.expand_dims(data_seeker.ccmpred, axis=2)
            # end_time = time.time()
            # LOGGER.info(f'CCmpred takes  {end_time-start_time}')
            if ccmpred.shape != evfold.shape:
                return
            data[FEATURES.evfold] = evfold
            data[FEATURES.ccmpred] = ccmpred
        except Exception as e:
            LOGGER.error(f'Data error for protein {protein}:\n{str(e)}')
            return
        try:
            # start_time = time.time()
            data[FEATURES.k_reference_dm_conv] = data_creator.k_reference_dm_test
            # end_time = time.time()
            # LOGGER.info(f'Refs dm takes  {end_time-start_time}')
            # start_time = time.time()
            data[FEATURES.seq_refs] = data_creator.seq_refs_test
            # end_time = time.time()
            # LOGGER.info(f'Refs seqs takes  {end_time-start_time}')
        except Exception as e:
            LOGGER.error(f'Data error for protein {protein}:\n{str(e)}')
            return

        # else:
        #     LOGGER.info('Using old alignemnt')
        #
        #     data[FEATURES.k_reference_dm_conv] = data_seeker.k_reference_dm_conv
        #     data[FEATURES.seq_refs] = data_creator.seq_refs
        data[FEATURES.seq_target] = data_creator.seq_target

        has_nones = False
        for f in data:
            has_nones |= data[f] is None
        if has_nones:
            return
        target_sequence_length = len(data_seeker.protein.sequence)
        data['sequence_length'] = np.array([target_sequence_length])

        if target_sequence_length > 1200:
            return

        if self._mode != tf.estimator.ModeKeys.PREDICT:
            cm = bin_array(data_seeker.protein.dm, self._num_bins)
            data['contact_map'] = cm if l is None else self._pad_feature(l, cm)

        if self._mode == tf.estimator.ModeKeys.PREDICT:
            self._yielded.append(protein)
        pkl_save(file, data)

        return data


class MsSsCCmpredPssmGenerator(DataGenerator):
    @property
    def required_shape_types(self):
        shapes = {
            FEATURES.seq_refs_pssm_ss: (None, PROTEIN_BOW_DIM_PSSM_SS, self._n_refs),
            FEATURES.seq_target_pssm_ss: (None, PROTEIN_BOW_DIM_PSSM_SS),
            FEATURES.evfold: (None, None, 1),
            FEATURES.k_reference_dm_conv: (None, None, self._n_refs),
            FEATURES.ccmpred: (None, None, 1)
        }

        if self._mode != tf.estimator.ModeKeys.PREDICT:
            shapes['contact_map'] = (None, None, 1)

        types = {f: tf.float32 for f in shapes}
        shapes['sequence_length'] = (1,)
        types['sequence_length'] = tf.int32

        return shapes, types

    def _yield_random_data(self, l):
        data = {}
        if self._mode != tf.estimator.ModeKeys.PREDICT:
            data['contact_map'] = np.array(
                np.random.randint(low=0, high=2, size=(l, l, 1)), np.float32)
        data[FEATURES.seq_refs_pssm_ss] = np.random.random(
            (l, PROTEIN_BOW_DIM_PSSM_SS, self._n_refs))
        data[FEATURES.seq_target_pssm_ss] = np.random.random((l, PROTEIN_BOW_DIM_PSSM_SS))
        data[FEATURES.evfold] = np.random.random((l, l, 1))
        data[FEATURES.ccmpred] = np.random.random((l, l, 1))
        data[FEATURES.k_reference_dm_conv] = np.random.random(
            (l, l, self._n_refs))
        data['sequence_length'] = np.array([l], dtype=np.int32)

        return data

    def _yield_protein_data(self, protein, l=None):
        data = {}

        data_seeker = DataSeeker(protein, n_refs=self._n_refs)
        data_creator = DataCreator(protein, self._n_refs)

        try:
            data[FEATURES.evfold] = np.expand_dims(data_seeker.evfold, axis=2)
            data[FEATURES.k_reference_dm_conv] = data_seeker.k_reference_dm_conv
            data[FEATURES.seq_refs_pssm_ss] = data_creator.seq_refs_pssm_ss
            data[FEATURES.seq_target_pssm_ss] = data_creator.seq_target_pssm_ss
            data[FEATURES.ccmpred] = np.expand_dims(data_seeker.ccmpred, axis=2)

        except (AttributeError, np.AxisError):
            return
        has_nones = False
        for f in data:
            has_nones |= data[f] is None
        if has_nones:
            return

        target_sequence_length = len(data_seeker.protein.sequence)
        data['sequence_length'] = np.array([target_sequence_length])

        if target_sequence_length > 1200:
            return

        if self._mode != tf.estimator.ModeKeys.PREDICT:
            cm = bin_array(data_seeker.protein.dm, self._num_bins)
            data['contact_map'] = cm if l is None else self._pad_feature(l, cm)

        if self._mode == tf.estimator.ModeKeys.PREDICT:
            self._yielded.append(protein)

        return data


class ConvGenerator(DataGenerator):
    @property
    def required_shape_types(self):
        shapes = {f: FEATURES_DIMS[f] for f in self._conv_features}
        types = {f: tf.float32 for f in self._conv_features}

        shapes['sequence_length'] = (1,)
        types['sequence_length'] = tf.int32

        if 'k_reference_dm' in shapes:
            shapes['k_reference_dm'] = (None, None, 1, self._n_refs)

        if 'k_reference_dm_conv' in shapes:
            shapes['k_reference_dm_conv'] = (None, None, self._n_refs)

        if self._mode != tf.estimator.ModeKeys.PREDICT:
            shapes['contact_map'] = (None, None, 1)
            types['contact_map'] = tf.float32

        return shapes, types

    def _yield_random_data(self, l):
        data = {}
        if self._mode != tf.estimator.ModeKeys.PREDICT:
            data['contact_map'] = np.array(
                np.random.randint(low=0, high=2, size=(l, l, 1)), np.float32)

        FEATURES_DIMS['k_reference_dm_conv'] = (None, None, self._n_refs)

        data.update({
            f: np.array(np.random.random([l, l, FEATURES_DIMS[f][2]]),
                        np.float32)
            for f in self._conv_features
        })
        data['sequence_length'] = np.array([l], dtype=np.int32)

        return data

    def _yield_protein_data(self, protein, l=None):

        data_seeker = DataSeeker(protein, n_refs=self._n_refs)
        data_creator = DataCreator(protein, n_refs=self._n_refs)

        data = {}

        for f in self._conv_features:

            feature = getattr(data_seeker, f) if f != FEATURES.k_reference_dm_conv else data_creator.k_reference_dm_test
            if feature is None:
                return
            if len(feature.shape) == 2:
                feature = np.expand_dims(feature, axis=2)
            data[f] = feature if l is None else self._pad_feature(
                l, feature)

        target_sequence_length = len(data_seeker.protein.sequence)
        data['sequence_length'] = np.array([target_sequence_length])

        if target_sequence_length > 1200:
            return

        if self._mode != tf.estimator.ModeKeys.PREDICT:
            cm = bin_array(data_seeker.protein.dm, self._num_bins)
            data['contact_map'] = cm if l is None else self._pad_feature(l, cm)

        if self._mode == tf.estimator.ModeKeys.PREDICT:
            self._yielded.append(protein)

        return data


class PeriscopeGenerator(DataGenerator):
    @property
    def required_shape_types(self):
        shapes = {
            FEATURES.seq_refs: (None, PROTEIN_BOW_DIM, self._n_refs),
            FEATURES.seq_target: (None, PROTEIN_BOW_DIM),
            FEATURES.evfold: (None, None, 1),
            FEATURES.k_reference_dm_conv: (None, None, self._n_refs),
            FEATURES.ccmpred: (None, None, 1),
            FEATURES.pwm_w: (None, 21),
            FEATURES.pwm_evo: (None, 21),
            FEATURES.conservation: (None, 1),
            FEATURES.beff: (1,)

        }

        if self._mode != tf.estimator.ModeKeys.PREDICT:
            dim = 1 if self._num_bins <= 2 else self._num_bins
            shapes['contact_map'] = (None, None, dim)

        types = {f: tf.float32 for f in shapes}
        shapes['sequence_length'] = (1,)
        types['sequence_length'] = tf.int32

        return shapes, types

    def _yield_random_data(self, l):
        data = {}
        if self._mode != tf.estimator.ModeKeys.PREDICT:
            data['contact_map'] = bin_array(np.array(
                np.random.randint(low=0, high=2, size=(l, l, 1)), np.float32), self._num_bins)
        data[FEATURES.seq_refs] = np.random.random(
            (l, PROTEIN_BOW_DIM, self._n_refs))
        data[FEATURES.seq_target] = np.random.random((l, PROTEIN_BOW_DIM))
        data[FEATURES.evfold] = np.random.random((l, l, 1))
        data[FEATURES.ccmpred] = np.random.random((l, l, 1))
        data[FEATURES.k_reference_dm_conv] = np.random.random(
            (l, l, self._n_refs))
        data[FEATURES.pwm_w] = np.random.random((l, 21))
        data[FEATURES.pwm_evo] = np.random.random((l, 21))
        data[FEATURES.conservation] = np.expand_dims(np.array(np.random.random((l)), dtype=np.float32), axis=1)
        data[FEATURES.beff] = np.array(np.random.random(1), dtype=np.float32)

        data['sequence_length'] = np.array([l], dtype=np.int32)

        return data

    def _yield_protein_data(self, protein, l=None):

        file = os.path.join(self._tmp_dir.name, f'{protein}.pkl')
        if os.path.isfile(file) and self._mode != tf.estimator.ModeKeys.PREDICT:
            LOGGER.info(f"loading {file}")
            return pkl_load(file)

        data = {}
        data_seeker = DataSeeker(protein, n_refs=self._n_refs)
        data_creator = DataCreator(protein, n_refs=self._n_refs)
        LOGGER.info(protein)

        try:
            # start_time = time.time()
            evfold = np.expand_dims(data_seeker.evfold, axis=2)
            # end_time = time.time()
            # LOGGER.info(f'Evfold takes  {end_time-start_time}')
            # start_time = time.time()
            ccmpred = np.expand_dims(data_seeker.ccmpred, axis=2)
            # end_time = time.time()
            # LOGGER.info(f'CCmpred takes  {end_time-start_time}')
            if ccmpred.shape != evfold.shape:
                return
            data[FEATURES.evfold] = evfold
            data[FEATURES.ccmpred] = ccmpred
        except Exception as e:
            LOGGER.error(f'Data error for protein {protein}:\n{str(e)}')
            return
        try:
            # start_time = time.time()
            data[FEATURES.k_reference_dm_conv] = data_creator.k_reference_dm_test
            # end_time = time.time()
            # LOGGER.info(f'Refs dm takes  {end_time-start_time}')
            # start_time = time.time()
            data[FEATURES.seq_refs] = data_creator.seq_refs_test
            data[FEATURES.pwm_w] = data_seeker.pwm_w
            data[FEATURES.pwm_evo] = data_seeker.pwm_evo
            data[FEATURES.conservation] = data_seeker.conservation
            data[FEATURES.beff] = data_seeker.beff

            # end_time = time.time()
            # LOGGER.info(f'Refs seqs takes  {end_time-start_time}')
        except Exception as e:
            LOGGER.error(f'Data error for protein {protein}:\n{str(e)}')
            return

        data[FEATURES.seq_target] = data_creator.seq_target

        has_nones = False
        for f in data:
            has_nones |= data[f] is None
        if has_nones:
            return
        target_sequence_length = len(data_seeker.protein.sequence)
        data['sequence_length'] = np.array([target_sequence_length])

        if target_sequence_length > 894:
            return

        if self._mode != tf.estimator.ModeKeys.PREDICT:
            cm = bin_array(data_seeker.protein.dm, self._num_bins)
            data['contact_map'] = cm if l is None else self._pad_feature(l, cm)

        if self._mode == tf.estimator.ModeKeys.PREDICT:
            self._yielded.append(protein)
        pkl_save(file, data)

        return data


class PeriscopeGeneratorSsAcc(DataGenerator):
    @property
    def required_shape_types(self):
        shapes = {
            FEATURES.seq_refs: (None, PROTEIN_BOW_DIM + 9, self._n_refs),
            # FEATURES.evfold: (None, None, 1),
            FEATURES.k_reference_dm_conv: (None, None, self._n_refs),
            FEATURES.ccmpred: (None, None, 1),
            FEATURES.pwm_w: (None, 21),
            FEATURES.pwm_evo: (None, 30),
            FEATURES.conservation: (None, 1),
            FEATURES.beff: (1,)

        }

        if self._mode != tf.estimator.ModeKeys.PREDICT:
            dim = 1 if self._num_bins <= 2 else self._num_bins
            shapes['contact_map'] = (None, None, dim)

        types = {f: tf.float32 for f in shapes}
        shapes['sequence_length'] = (1,)
        types['sequence_length'] = tf.int32

        return shapes, types

    def _yield_random_data(self, l):
        data = {}
        if self._mode != tf.estimator.ModeKeys.PREDICT:
            data['contact_map'] = bin_array(np.array(
                np.random.randint(low=0, high=2, size=(l, l)), np.float32), self._num_bins)
        data[FEATURES.seq_refs] = np.random.random(
            (l, PROTEIN_BOW_DIM + 9, self._n_refs))
        # data[FEATURES.evfold] = np.random.random((l, l, 1))
        data[FEATURES.ccmpred] = np.random.random((l, l, 1))
        data[FEATURES.k_reference_dm_conv] = np.random.random(
            (l, l, self._n_refs))
        data[FEATURES.pwm_w] = np.random.random((l, 21))
        data[FEATURES.pwm_evo] = np.random.random((l, 30))
        data[FEATURES.conservation] = np.expand_dims(np.array(np.random.random((l)), dtype=np.float32), axis=1)
        data[FEATURES.beff] = np.array(np.random.random(1), dtype=np.float32)

        data['sequence_length'] = np.array([l], dtype=np.int32)

        return data

    def _yield_protein_data(self, protein, l=None):
        drop = np.random.binomial(1, self._templates_dropout)

        file = os.path.join(self._tmp_dir.name, f'{protein}.pkl')
        if os.path.isfile(file) and self._mode == tf.estimator.ModeKeys.TRAIN:
            LOGGER.info(f"loading {file}")
            data = pkl_load(file)
            if drop == 1:
                LOGGER.info('Templates dropped')
                data[FEATURES.k_reference_dm_conv] = np.zeros_like(data[FEATURES.k_reference_dm_conv])
            return data

        data = {}

        LOGGER.info(protein)

        try:
            data_seeker = DataSeeker(protein, n_refs=self._n_refs)
            data_creator = DataCreator(protein, n_refs=self._n_refs, family=self._family,
                                       require_template=self._require_template)

            if not data_creator.has_msa:
                return

            # start_time = time.time()
            # evfold = np.expand_dims(data_seeker.evfold, axis=2)
            # end_time = time.time()
            # LOGGER.info(f'Evfold takes  {end_time-start_time}')
            # start_time = time.time()
            ccmpred = np.expand_dims(data_creator.ccmpred, axis=2)
            # end_time = time.time()
            # LOGGER.info(f'CCmpred takes  {end_time-start_time}')
            # if ccmpred.shape != evfold.shape:
            #     return
            # data[FEATURES.evfold] = evfold
            data[FEATURES.ccmpred] = ccmpred
        except Exception as e:
            LOGGER.error(f'Data error for protein {protein}:\n{str(e)}')
            return
        try:
            data[FEATURES.k_reference_dm_conv] = data_creator.k_reference_dm_test
            data[FEATURES.seq_refs] = data_creator.seq_refs_ss_acc

            if drop == 1 and self._mode == tf.estimator.ModeKeys.TRAIN:
                LOGGER.info('Templates dropped')
                data[FEATURES.k_reference_dm_conv] = np.zeros_like(data_creator.k_reference_dm_test)

            data[FEATURES.pwm_w] = data_creator.pwm_w
            data[FEATURES.pwm_evo] = data_creator.pwm_evo_ss
            data[FEATURES.conservation] = data_creator.conservation
            data[FEATURES.beff] = data_creator.beff

        except Exception as e:
            LOGGER.error(f'Data error for protein {protein}:\n{str(e)}')
            return

        has_nones = False
        for f in data:
            has_nones |= data[f] is None
        if has_nones:
            return
        target_sequence_length = len(data_creator.str_seq)
        data['sequence_length'] = np.array([target_sequence_length])

        if target_sequence_length >= 650:
            return

        if self._mode != tf.estimator.ModeKeys.PREDICT:
            cm = bin_array(data_seeker.protein.dm, self._num_bins)
            data['contact_map'] = cm if l is None else self._pad_feature(l, cm)

        if self._mode == tf.estimator.ModeKeys.PREDICT:
            self._yielded.append(protein)
        if drop:
            data[FEATURES.k_reference_dm_conv] = data_creator.k_reference_dm_test
        pkl_save(file, data)

        return data


class PeriscopePropertiesGenerator(DataGenerator):
    @property
    def required_shape_types(self):
        shapes = {
            FEATURES.seq_refs: (None, PROTEIN_BOW_DIM + 9, self._n_refs),
            # FEATURES.evfold: (None, None, 1),
            FEATURES.k_reference_dm_conv: (None, None, self._n_refs),
            FEATURES.ccmpred: (None, None, 1),
            FEATURES.pwm_w: (None, 21),
            FEATURES.pwm_evo: (None, 30),
            FEATURES.conservation: (None, 1),
            FEATURES.beff: (1,),
            FEATURES.seq_target: (None, PROTEIN_BOW_DIM),
            FEATURES.properties_target: (None, 13)

        }

        if self._mode != tf.estimator.ModeKeys.PREDICT:
            dim = 1 if self._num_bins <= 2 else self._num_bins
            shapes['contact_map'] = (None, None, dim)

        types = {f: tf.float32 for f in shapes}
        shapes['sequence_length'] = (1,)
        types['sequence_length'] = tf.int32

        return shapes, types

    def _yield_random_data(self, l):
        data = {}
        if self._mode != tf.estimator.ModeKeys.PREDICT:
            data['contact_map'] = bin_array(np.array(
                np.random.randint(low=0, high=2, size=(l, l)), np.float32), self._num_bins)
        data[FEATURES.seq_refs] = np.random.random(
            (l, PROTEIN_BOW_DIM + 9, self._n_refs))
        # data[FEATURES.evfold] = np.random.random((l, l, 1))
        data[FEATURES.ccmpred] = np.random.random((l, l, 1))
        data[FEATURES.k_reference_dm_conv] = np.random.random(
            (l, l, self._n_refs))
        data[FEATURES.pwm_w] = np.random.random((l, 21))
        data[FEATURES.pwm_evo] = np.random.random((l, 30))
        data[FEATURES.conservation] = np.expand_dims(np.array(np.random.random((l)), dtype=np.float32), axis=1)
        data[FEATURES.beff] = np.array(np.random.random(1), dtype=np.float32)
        data[FEATURES.properties_target] = np.random.random((l, 13))
        data[FEATURES.seq_target] = np.random.random((l,22))

        data['sequence_length'] = np.array([l], dtype=np.int32)

        return data

    def _yield_protein_data(self, protein, l=None):
        start = time.time()
        drop = np.random.binomial(1, self._templates_dropout)

        file = os.path.join(self._tmp_dir.name, f'{protein}.pkl')
        if os.path.isfile(file) and self._mode == tf.estimator.ModeKeys.TRAIN:
            LOGGER.info(f"loading {file}")
            data = pkl_load(file)
            if drop == 1:
                LOGGER.info('Templates dropped')
                data[FEATURES.k_reference_dm_conv] = np.zeros_like(data[FEATURES.k_reference_dm_conv])
            return data

        data = {}

        LOGGER.info(protein)

        try:
            # start_time = time.time()

            # data_seeker = DataSeeker(protein, n_refs=self._n_refs)
            data_creator = DataCreator(protein, n_refs=self._n_refs, family=self._family,
                                       require_template=self._require_template)
            # end_time = time.time()
            # LOGGER.info(f'creator takes  {end_time-start_time}')
            if not data_creator.has_refs:
                return

            # start_time = time.time()
            # evfold = np.expand_dims(data_seeker.evfold, axis=2)
            # end_time = time.time()
            # LOGGER.info(f'Evfold takes  {end_time-start_time}')
            # start_time = time.time()
            ccmpred = np.expand_dims(data_creator.ccmpred, axis=2)
            # end_time = time.time()
            # LOGGER.info(f'CCmpred takes  {end_time-start_time}')
            # if ccmpred.shape != evfold.shape:
            #     return
            # data[FEATURES.evfold] = evfold
            data[FEATURES.ccmpred] = ccmpred
        except Exception as e:
            LOGGER.error(f'Data error for protein {protein}:\n{str(e)}')
            return
        try:
            # start_time = time.time()

            data[FEATURES.k_reference_dm_conv] = data_creator.k_reference_dm_test
            # end_time = time.time()
            # LOGGER.info(f'k_reference_dm_test takes  {end_time-start_time}')
            # start_time = time.time()

            data[FEATURES.seq_refs] = data_creator.seq_refs_ss_acc
            # end_time = time.time()
            # LOGGER.info(f'seq_refs_ss_acc takes  {end_time-start_time}')

            if drop == 1 and self._mode == tf.estimator.ModeKeys.TRAIN:
                LOGGER.info('Templates dropped')
                data[FEATURES.k_reference_dm_conv] = np.zeros_like(data_creator.k_reference_dm_test)
            # start_time = time.time()

            data[FEATURES.pwm_w] = data_creator.pwm_w
            data[FEATURES.pwm_evo] = data_creator.pwm_evo_ss
            data[FEATURES.conservation] = data_creator.conservation
            data[FEATURES.beff] = data_creator.beff
            # end_time = time.time()
            # LOGGER.info(f'scores takes  {end_time-start_time}')
            # start_time = time.time()

            data[FEATURES.properties_target] = data_creator.raptor_properties
            # end_time = time.time()
            # LOGGER.info(f'raptor_properties takes  {end_time-start_time}')
            data[FEATURES.seq_target] = data_creator.seq_target

        except Exception as e:
            LOGGER.error(f'Data error for protein {protein}:\n{str(e)}')
            return

        has_nones = False
        for f in data:
            has_nones |= data[f] is None
        if has_nones:
            return
        target_sequence_length = len(data_creator.str_seq)
        data['sequence_length'] = np.array([target_sequence_length])

        if target_sequence_length >= 650:
            return

        if self._mode != tf.estimator.ModeKeys.PREDICT:
            cm = bin_array(data_creator.protein.dm, self._num_bins)
            data['contact_map'] = cm if l is None else self._pad_feature(l, cm)

        if self._mode == tf.estimator.ModeKeys.PREDICT:
            self._yielded.append(protein)
        if drop:
            data[FEATURES.k_reference_dm_conv] = data_creator.k_reference_dm_test
        pkl_save(file, data)
        LOGGER.info(f"Time elapsed {time.time() - start}")
        return data


class TemplatesGenerator(DataGenerator):
    @property
    def required_shape_types(self):
        shapes = {
            FEATURES.seq_refs: (None, PROTEIN_BOW_DIM + 9, self._n_refs),
            FEATURES.k_reference_dm_conv: (None, None, self._n_refs),
            FEATURES.pwm_w: (None, 21),

        }

        if self._mode != tf.estimator.ModeKeys.PREDICT:
            dim = 1 if self._num_bins <= 2 else self._num_bins
            shapes['contact_map'] = (None, None, dim)

        types = {f: tf.float32 for f in shapes}
        shapes['sequence_length'] = (1,)
        types['sequence_length'] = tf.int32

        return shapes, types

    def _yield_random_data(self, l):
        data = {}
        if self._mode != tf.estimator.ModeKeys.PREDICT:
            data['contact_map'] = bin_array(np.array(
                np.random.randint(low=0, high=2, size=(l, l, 1)), np.float32), self._num_bins)
        data[FEATURES.seq_refs] = np.random.random(
            (l, PROTEIN_BOW_DIM + 9, self._n_refs))
        data[FEATURES.k_reference_dm_conv] = np.random.random(
            (l, l, self._n_refs))
        data[FEATURES.pwm_w] = np.random.random((l, 21))

        data['sequence_length'] = np.array([l], dtype=np.int32)

        return data

    def _yield_protein_data(self, protein, l=None):

        file = os.path.join(self._tmp_dir.name, f'{protein}.pkl')
        if os.path.isfile(file) and self._mode != tf.estimator.ModeKeys.PREDICT:
            LOGGER.info(f"loading {file}")
            data = pkl_load(file)

            return data

        data = {}
        data_seeker = DataSeeker(protein, n_refs=self._n_refs)
        data_creator = DataCreator(protein, n_refs=self._n_refs)
        LOGGER.info(protein)

        try:
            data[FEATURES.k_reference_dm_conv] = data_creator.k_reference_dm_test
            data[FEATURES.seq_refs] = data_creator.seq_refs_ss_acc
            data[FEATURES.pwm_w] = data_seeker.pwm_w

        except Exception as e:
            LOGGER.error(f'Data error for protein {protein}:\n{str(e)}')
            return

        has_nones = False
        for f in data:
            has_nones |= data[f] is None
        if has_nones:
            return
        target_sequence_length = len(data_seeker.protein.sequence)
        data['sequence_length'] = np.array([target_sequence_length])

        if target_sequence_length >= 650 and self._mode != tf.estimator.ModeKeys.PREDICT:
            return

        if self._mode != tf.estimator.ModeKeys.PREDICT:
            cm = bin_array(data_seeker.protein.dm, self._num_bins)
            data['contact_map'] = cm if l is None else self._pad_feature(l, cm)

        if self._mode == tf.estimator.ModeKeys.PREDICT:
            self._yielded.append(protein)

        pkl_save(file, data)

        return data


class EvoGenerator(DataGenerator):
    @property
    def required_shape_types(self):
        shapes = {
            FEATURES.evfold: (None, None, 1),
            FEATURES.ccmpred: (None, None, 1),
            FEATURES.pwm_evo: (None, 21),
            FEATURES.conservation: (None, 1),
            FEATURES.beff: (1,)

        }

        if self._mode != tf.estimator.ModeKeys.PREDICT:
            dim = 1 if self._num_bins <= 2 else self._num_bins
            shapes['contact_map'] = (None, None, dim)

        types = {f: tf.float32 for f in shapes}
        shapes['sequence_length'] = (1,)
        types['sequence_length'] = tf.int32

        return shapes, types

    def _yield_random_data(self, l):
        data = {}
        if self._mode != tf.estimator.ModeKeys.PREDICT:
            data['contact_map'] = bin_array(np.array(
                np.random.randint(low=0, high=2, size=(l, l, 1)), np.float32), self._num_bins)

        data[FEATURES.evfold] = np.random.random((l, l, 1))
        data[FEATURES.ccmpred] = np.random.random((l, l, 1))

        data[FEATURES.pwm_evo] = np.random.random((l, 21))
        data[FEATURES.conservation] = np.expand_dims(np.array(np.random.random((l)), dtype=np.float32), axis=1)
        data[FEATURES.beff] = np.array(np.random.random(1), dtype=np.float32)

        data['sequence_length'] = np.array([l], dtype=np.int32)

        return data

    def _yield_protein_data(self, protein, l=None):

        file = os.path.join(self._tmp_dir.name, f'{protein}.pkl')
        if os.path.isfile(file) and self._mode != tf.estimator.ModeKeys.PREDICT:
            LOGGER.info(f"loading {file}")
            data = pkl_load(file)
            return data

        data = {}
        data_seeker = DataSeeker(protein, n_refs=self._n_refs)
        LOGGER.info(protein)

        try:
            # start_time = time.time()
            evfold = np.expand_dims(data_seeker.evfold, axis=2)
            # end_time = time.time()
            # LOGGER.info(f'Evfold takes  {end_time-start_time}')
            # start_time = time.time()
            ccmpred = np.expand_dims(data_seeker.ccmpred, axis=2)
            # end_time = time.time()
            # LOGGER.info(f'CCmpred takes  {end_time-start_time}')
            if ccmpred.shape != evfold.shape:
                return
            data[FEATURES.evfold] = evfold
            data[FEATURES.ccmpred] = ccmpred
        except Exception as e:
            LOGGER.error(f'Data error for protein {protein}:\n{str(e)}')
            return
        try:

            data[FEATURES.pwm_evo] = data_seeker.pwm_evo
            data[FEATURES.conservation] = data_seeker.conservation
            data[FEATURES.beff] = data_seeker.beff

        except Exception as e:
            LOGGER.error(f'Data error for protein {protein}:\n{str(e)}')
            return

        has_nones = False
        for f in data:
            has_nones |= data[f] is None
        if has_nones:
            return
        target_sequence_length = len(data_seeker.protein.sequence)
        data['sequence_length'] = np.array([target_sequence_length])

        if target_sequence_length >= 650:
            return

        if self._mode != tf.estimator.ModeKeys.PREDICT:
            cm = bin_array(data_seeker.protein.dm, self._num_bins)
            data['contact_map'] = cm if l is None else self._pad_feature(l, cm)

        if self._mode == tf.estimator.ModeKeys.PREDICT:
            self._yielded.append(protein)

        pkl_save(file, data)

        return data


Generators = {ARCHS.conv: ConvGenerator,
              ARCHS.ms_ss_ccmpred: MsSsCCmpredGenerator,
              ARCHS.ms_ss_ccmpred_pssm: MsSsCCmpredPssmGenerator,
              ARCHS.multi_structure_ccmpred: MsCCmpredGenerator,
              ARCHS.multi_structure_ccmpred_2: MsCCmpredGenerator,
              ARCHS.periscope: PeriscopeGenerator,
              ARCHS.periscope2: PeriscopeGeneratorSsAcc,
              ARCHS.evo: EvoGenerator,
              ARCHS.templates: TemplatesGenerator,
              ARCHS.periscope_properties: PeriscopePropertiesGenerator}
