"""Protein data seeker

This class is in charge of seeking data that is already in memory

"""
import os
import logging

import numpy as np

from ..utils.constants import PATHS, DATASETS, SEQ_ID_THRES, N_REFS
from ..utils.protein import Protein
from ..utils.utils import (get_sotred_ec, read_raw_ec_file, pkl_save, MODELLER_VERSION, pkl_load, get_modeller_pdb_file,
                           get_target_path, get_target_ccmpred_file, get_target_scores_file)

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DataSeeker:
    _MSA_VERSION = 2
    _EVFOLD_VERSION = 2
    _PSSM_VERSION = 2
    NAN_VALUE = -1.0
    SEQUENCE_LENGTH_THRESHOLD = 50
    _MODELLER_VERSION = MODELLER_VERSION
    _STRUCTURES_VERSION = 3
    _THRESHOLD = 8

    def __init__(self, protein, n_refs=N_REFS):
        self.protein = Protein(protein[0:4], protein[4])
        self.target = self.protein.target
        self._msa_data_path = os.path.join(get_target_path(protein), 'features')
        self._n_refs = n_refs

    def _get_closest_reference(self):

        if self.sorted_structures is None:
            return
        structure = self.sorted_structures.index[0]
        return structure

    @property
    def closest_reference(self):
        return self._get_closest_reference()

    @property
    def sorted_structures(self):
        structures_sorted_file = os.path.join(
            self._msa_data_path, 'structures_sorted.pkl')

        sorted_structures = pkl_load(structures_sorted_file)

        if sorted_structures is None:
            return

        struc_dm = self._get_phylo_structures_mat()

        if struc_dm is None:
            return

        struc_dm = struc_dm.loc[self.target, :]
        struc_dm_valid = struc_dm[struc_dm > 1 - SEQ_ID_THRES].index

        sorted_structures = sorted_structures[struc_dm_valid].sort_values(ascending=False)
        if len(sorted_structures) == 0:
            return
        return sorted_structures

    def _get_phylo_structures_mat(self):

        version = self._STRUCTURES_VERSION

        phylo_structures_file = os.path.join(
            self._msa_data_path, 'structures_phylo_dist_mat_v%s.pkl' % version)

        if os.path.isfile(phylo_structures_file):
            return pkl_load(phylo_structures_file)
        return

    @property
    def total_refs(self):
        if self.sorted_structures is None:
            return 0

        return len(self.sorted_structures)

    def _get_structure_file(self, homologous, version, mean=False):
        if mean:
            os.path.join(self._msa_data_path, f'{homologous}_mean_v{version}.pkl')
        return os.path.join(self._msa_data_path, f'{homologous}_structure_v{version}.pkl')

    def _get_ref_dm(self, reference):

        version = self._STRUCTURES_VERSION
        dm_file = self._get_structure_file(reference, version)
        return pkl_load(dm_file)

    def _get_pssm(self):
        version = self._MSA_VERSION

        pssm_file = os.path.join(self._msa_data_path, f'pssm_bio_v{version}.pkl')

        return pkl_load(pssm_file)

    def _get_aligned_ss(self):

        ss_refs_file = os.path.join(self._msa_data_path, 'ss_refs.pkl')
        return pkl_load(ss_refs_file)

    def _get_ss_refs(self):
        ss_refs_full = self._get_aligned_ss()

        if ss_refs_full is None:
            return

        shape = ss_refs_full.shape
        total_refs = shape[-1]

        if total_refs >= self._n_refs:
            return ss_refs_full[..., 0:self._n_refs]
        else:
            return np.concatenate(
                [np.zeros((shape[0], shape[1], self._n_refs - total_refs)), ss_refs_full],
                axis=2)

    def _validate_ec(self, raw_ec):
        # validation that the sequences match
        raw_ec_local = raw_ec.copy()

        ec_sorted_i, ec_sorted_j, ec = get_sotred_ec(raw_ec_local)

        # ec shows no duplicates hence i goes until on aa before the last
        last_msa_string = ec_sorted_j.A_j.iloc[-1]

        msa_seq = "".join(list(ec_sorted_i.A_i) + list(last_msa_string))
        is_valid = msa_seq == "".join(list(self.protein.sequence))
        return is_valid

    def _get_closest_references(self):
        sorted_structures_by_distance = self.sorted_structures

        if sorted_structures_by_distance is None:
            return
        structures = sorted_structures_by_distance.sort_values(ascending=True)

        return list(structures.index)

    def get_average_modeller_dm(self, n_structures):
        """Returns the average cm over few modeller predicted structures

        Args:
            n_structures (int): number of modeller predicted structures

        Returns:
            np.array: of shape (l, l)

        """

        cms = []
        test_sets = set(DATASETS.pfam) | set(DATASETS.membrane) | set(DATASETS.cameo)

        if self.closest_reference is None or self.protein.target not in test_sets:
            return

        for n_struc in range(1, n_structures + 1):
            pdb_file_path = get_modeller_pdb_file(target=self.protein.target, n_struc=n_struc)
            has_modeller_file = os.path.isfile(pdb_file_path)
            if not has_modeller_file:
                LOGGER.info(f'shite, not modeller for {self.protein.target}')
            modeller_prot = Protein(self.protein.protein,
                                    self.protein.chain,
                                    pdb_path=PATHS.modeller,
                                    modeller_n_struc=n_struc)
            modeller_dm = modeller_prot.dm
            assert len(modeller_prot.sequence) == len(self.protein.sequence)

            # thres = 8 if version == 4 else 8.5
            #
            cms.append(modeller_dm)

        return np.nanmean(np.stack(cms, axis=2), axis=2)

    @property
    def target_pdb_dm(self):
        return self.protein.dm

    @property
    def target_pdb_cm(self):
        return self._get_cm(self.target_pdb_dm)

    @property
    def ccmpred(self):

        ccmpred_mat_file = get_target_ccmpred_file(self.target)
        if os.path.isfile(ccmpred_mat_file):
            ccmpred_mat = np.loadtxt(ccmpred_mat_file)
            if ccmpred_mat.shape[0] != len(self.protein.str_seq):
                return
            return ccmpred_mat

        return

    @property
    def evfold(self):
        version = self._EVFOLD_VERSION

        evfold_file = '%s_v%s.txt' % (self.protein.target, version)
        evfold_mat_file = '%s_v%s.pkl' % (self.protein.target, version)
        evfold_path = os.path.join(get_target_path(self.target), 'evfold')

        evfold_mat_path = os.path.join(evfold_path, evfold_mat_file)

        evfold_mat = pkl_load(evfold_mat_path)
        if evfold_mat is not None:
            if evfold_mat.shape[0] != len(self.protein.str_seq):
                return
            return evfold_mat

        ec_file = os.path.join(evfold_path, evfold_file)
        if not os.path.isfile(ec_file):
            return

        raw_ec = read_raw_ec_file(ec_file)
        is_valid = self._validate_ec(raw_ec)
        if not is_valid:
            return

        ec_sorted_i, ec_sorted_j, ec = get_sotred_ec(raw_ec)
        sorted_msa_indices = ec_sorted_i.i.to_list() + ec_sorted_j.j.to_list()
        sorted_msa_indices.sort()

        msa_ind_to_pdb_ind_map = {
            msa: pdb
            for pdb, msa in enumerate(set(sorted_msa_indices))
        }

        pdb_i = ec.loc[:, 'i'].map(msa_ind_to_pdb_ind_map).copy()
        pdb_j = ec.loc[:, 'j'].map(msa_ind_to_pdb_ind_map).copy()

        target_indices_pdb = (np.array(pdb_i).astype(np.int64),
                              np.array(pdb_j).astype(np.int64))

        l = len(self.protein.sequence)

        evfold = np.zeros(shape=(l, l))

        evfold[target_indices_pdb] = ec['cn'].values

        pkl_save(data=evfold, filename=evfold_mat_path)

        return evfold

    def _replace_nas(self, array):
        return np.nan_to_num(array, nan=self.NAN_VALUE)

    @property
    def reference_dm(self):

        reference = self._get_closest_reference()

        if reference is None:
            return

        ref_dm = self._get_ref_dm(reference)
        return self._replace_nas(ref_dm)

    def trim_pad_arr(self, arr):
        if arr is None:
            return
        n_strucs = int(arr.shape[-1])
        if n_strucs < self._n_refs:
            shape = list(arr.shape)
            shape[-1] = self._n_refs - n_strucs
            zero_array = np.zeros(shape)

            arr_out = np.concatenate([zero_array, arr], axis=2)
        if n_strucs >= self._n_refs:
            arr_out = arr[..., (n_strucs - self._n_refs): n_strucs]
        return self._replace_nas(arr_out)
    @property
    def k_reference_dm(self):

        dms = []
        structures = self._get_closest_references()
        if structures is None:
            return
        n_structures = len(structures)

        structures = structures if self._n_refs >= n_structures else structures[
                                                                     n_structures - self._n_refs:n_structures]

        for i in range(min(self._n_refs, n_structures)):
            s = structures[i]

            dm = self._get_ref_dm(reference=s)
            if dm is None:
                return None
            dm = dm if len(dm.shape) == 3 else np.expand_dims(dm, 2)

            dms.append(dm)
        if n_structures < self._n_refs:
            zero_array = np.zeros_like(dms[0])
            masking = [zero_array] * (self._n_refs - n_structures)
            dms = masking + dms
        k_refs = np.stack(dms, axis=-1)
        return self._replace_nas(k_refs)

    @property
    def k_reference_dm_conv(self):
        if self.k_reference_dm is None:
            return
        return np.squeeze(self.k_reference_dm)

    @property
    def seq_target(self):
        seq_target_file = os.path.join(self._msa_data_path, 'target_seq.pkl')

        if os.path.isfile(seq_target_file):
            return np.squeeze(pkl_load(seq_target_file))

    @property
    def seq_refs(self):

        seq_refs_file = os.path.join(self._msa_data_path, 'refs_seqs.pkl')

        refs_seqs = pkl_load(seq_refs_file)

        if refs_seqs is None:
            return

        shape = refs_seqs.shape
        total_refs = shape[-1]

        if total_refs >= self._n_refs:
            return refs_seqs[..., 0:self._n_refs]
        else:
            return np.concatenate(
                [np.zeros((shape[0], shape[1], self._n_refs - total_refs)), refs_seqs],
                axis=2)

    @property
    def seq_refs_pssm(self):

        pssm = self._get_pssm()
        refs = self.seq_refs

        if pssm is None or refs is None:
            return

        return np.array(
            np.concatenate([refs, np.repeat(np.expand_dims(pssm, axis=2), self._n_refs, 2)],
                           axis=1), dtype=np.float32)

    @property
    def seq_target_pssm(self):

        pssm = self._get_pssm()
        if pssm is None:
            return

        return np.array(np.concatenate([self.seq_target, pssm], axis=1), dtype=np.float32)

    @property
    def seq_refs_ss(self):

        ss_refs = self._get_ss_refs()
        refs = self.seq_refs

        if ss_refs is None:
            return

        return np.concatenate([ss_refs, refs], axis=1)

    @property
    def seq_target_ss(self):

        target_ss = self.protein.secondary_structure

        if target_ss is None:
            return

        return np.concatenate([target_ss, self.seq_target], axis=1)

    @property
    def seq_refs_pssm_ss(self):

        ss_refs = self._get_ss_refs()
        seq_refs_pssm = self.seq_refs_pssm
        if ss_refs is None or seq_refs_pssm is None:
            return

        return np.concatenate([ss_refs, seq_refs_pssm], axis=1)

    @property
    def seq_target_pssm_ss(self):

        target_ss = self.protein.secondary_structure

        seq_target_pssm = self.seq_target_pssm

        if seq_target_pssm is None or target_ss is None:
            return

        np.concatenate([self.protein.secondary_structure, self.seq_target_pssm], axis=1)

    @property
    def modeller_dm(self):

        pdb_file_path = get_modeller_pdb_file(target=self.protein.target, n_struc=1)

        if not os.path.isfile(pdb_file_path):
            return

        modeller_prot = Protein(self.protein.target[0:4],
                                self.protein.target[4],
                                pdb_path=PATHS.modeller,
                                version=self._STRUCTURES_VERSION,
                                modeller_n_struc=1)

        return modeller_prot.dm

    def _get_cm(self, dm):
        if dm is None:
            return
        nan_inds = np.logical_or(np.isnan(dm), dm == self.NAN_VALUE)
        dm[nan_inds] = self.NAN_VALUE

        cm = (dm < self._THRESHOLD).astype(np.float32)

        cm[nan_inds] = self.NAN_VALUE

        return cm

    @property
    def reference_cm(self):
        return self._get_cm(self.reference_dm)

    @property
    def modeller_cm(self):
        return self._get_cm(self.modeller_dm)

    @property
    def sequence_distance_matrix(self):
        def i_minus_j(i, j):
            return np.abs(i - j)

        sequence_distance = np.fromfunction(i_minus_j,
                                            shape=self.target_pdb_cm.shape)
        return sequence_distance

    @property
    def scores(self):
        scores_file = get_target_scores_file(self.target)

        return pkl_load(scores_file)

    @property
    def pwm_w(self):
        if self.scores is None:
            return
        return self.scores['PWM'][0]

    @property
    def pwm_evo(self):
        if self.scores is None:
            return
        return self.scores['PWM'][1]

    @property
    def conservation(self):
        if self.scores is None:
            return
        return np.expand_dims(np.array(self.scores['conservation']), axis=1)

    @property
    def beff(self):
        if self.scores is None:
            return
        return np.array([self.scores['Beff_alignment']], dtype=np.float32)