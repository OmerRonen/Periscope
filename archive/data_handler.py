import collections
import logging
import os
import warnings
import tempfile

from Bio.Align import MultipleSeqAlignment, AlignInfo
from Bio.Alphabet import Gapped
from Bio.Alphabet.IUPAC import IUPACProtein, ExtendedIUPACProtein
from deepdiff import DeepDiff
from Bio import SeqIO
from Bio.Seq import Seq
from scipy.special import softmax

from .globals import (MSA_STRUCTURES_DATA_PATH, HHBLITS_PATH, EVFOLD_PATH, MSA_DATA_PATH, MODELLER_PATH, DATASETS,
                      PROTEIN_BOW_DIM, NUM_HOMOLOGOUS, periscope_path, AMINO_ACID_STATS, CCMPRED_PATH)
from os import path
import pandas as pd
import numpy as np
from Bio.SeqRecord import SeqRecord
import subprocess
import urllib
import itertools
from .protein import Protein
import datetime

from .utils_old import (pkl_load, create_sifts_mapping, pkl_save, compute_pssm,
                   compute_structures_identity_matrix,
                   MODELLER_VERSION, get_modeller_pdb_file,VERSION,
                   get_local_sequence_identity_distmat, write_fasta, convert_to_aln)

logging.basicConfig(level=logging.INFO)

LOGGER = logging.getLogger(__name__)

SIFTS_MAPPING = create_sifts_mapping()
TEST_SET = DATASETS['pfam']

def read_fasta(filename):
    return list(SeqIO.parse(open(filename, "r"), "fasta"))[0]

def _get_uniprot_seq(uniprot):
    baseUrl = "http://www.uniprot.org/uniprot/"
    currentUrl = baseUrl + uniprot + ".fasta"
    file = 'tmp2.fasta'
    cmd = f'wget {currentUrl} -q -O {file}'


    subprocess.call(cmd, shell=True)

    Seq = read_fasta(file)
    os.remove(file)
    return Seq
def read_raw_ec_file(filename, sort=True, score="cn"):
    """
    Read a raw EC file (e.g. from plmc) and sort
    by scores

    Parameters
    ----------
    filename : str
        File containing evolutionary couplings
    sort : bool, optional (default: True)
        If True, sort pairs by coupling score in
        descending order
    score_column : str, optional (default: True)
        Score column to be used for sorting

    Returns
    -------
    ecs : pd.DataFrame
        Table of evolutionary couplings

    """
    ecs = pd.read_csv(filename,
                      sep=" ",
                      names=["i", "A_i", "j", "A_j", "fn", "cn"])

    if sort:
        ecs = ecs.sort_values(by=score, ascending=False)

    return ecs


class ProteinDataHandler:
    _VERSION = VERSION

    _MSA_VERSION = 2
    _PLMC_VERSION = 2
    _PSSM_VERSION = 2
    NAN_VALUE = -1.0
    SEQUENCE_LENGTH_THRESHOLD = 50
    _MODELLER_VERSION = MODELLER_VERSION
    _LOCAL_DISTANCE = 5

    def __init__(self, target, k=None, include_local_dist=False, mode="ask", structures_version=3, log_pssm=True):
        self._STRUCTURES_VERSION = structures_version
        self.THRESHOLD = 8 if self._STRUCTURES_VERSION == 3 else 8.5

        self.target = target
        self._mode = mode
        self._include_local_dist = include_local_dist
        self._protein_name, self._chain = target[0:4], target[4]
        self.protein = Protein(self._protein_name, self._chain, version=self._STRUCTURES_VERSION)
        if len(self.protein.sequence) < self.SEQUENCE_LENGTH_THRESHOLD:
            return
        self.msa_data_path = path.join(MSA_STRUCTURES_DATA_PATH, target)

        self._init_m_data = self._get_metadata()
        self._metadata = self._get_metadata()
        if self._metadata['version'] == 22:
            self._metadata['version'] = 2.2

        has_struc_data = self._has_structure_data()

        if not has_struc_data:
            self._gen_structures_version()

        target_dm = self.protein.dm

        self.target_pdb_dm, self.target_pdb_cm = self._gen_cm_dm(target_dm)
        self.target_outer_angels = self._get_angles_outer_product(
            self.protein.angels)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.reference_std = self._get_cm_dm_std()

        self.pssm = self._get_bio_pssm() if log_pssm else np.exp(-1 * self._get_bio_pssm())
        self._ec = self._get_ec()

        if self._ec is None:
            return

        self.plmc_score = self._get_plmc_score()
        self.ccmpred = self._get_ccmpred()
        if self.ccmpred is not None:
            assert self.plmc_score.shape[0] == self.ccmpred.shape[0] == len(self.protein.sequence)

        self.reference_dm, self.reference_cm = self._get_reference_dm_cm()
        if self._include_local_dist and self.reference_dm is not None:
            loc_dist_ref = self._get_local_dist_mat(
                reference=self.closest_known_strucutre)
            if loc_dist_ref is not None:
                self.reference_dm = np.stack([self.reference_dm, loc_dist_ref],
                                             axis=2)
            else:
                self.reference_dm = None
        if k is not None:
            self.k_reference_dm, self.k_reference_cm = self._get_k_reference_dm_cm(
                k)

            self.k_reference_dm_conv = np.squeeze(
                self.k_reference_dm
            ) if self.k_reference_dm is not None else None

            self.seq_refs = self._get_seq_refs(k=k)
            if self.seq_refs is None:
                return
            ss_refs = self._get_ss_refs(k=k)

            self.seq_refs_pssm = np.array(
                np.concatenate([self.seq_refs, np.repeat(np.expand_dims(self.pssm, axis=2), k, 2)],
                               axis=1), dtype=np.float32)

            self.seq_refs_ss = np.concatenate([ss_refs, self.seq_refs], axis=1) if ss_refs is not None else None

            self.seq_refs_pssm_ss = np.concatenate([ss_refs, self.seq_refs_pssm],
                                                   axis=1) if ss_refs is not None else None

            self.seq_target = self._get_seq_target()

            self.seq_target_pssm = np.array(np.concatenate([self.seq_target, self.pssm], axis=1), dtype=np.float32)

            has_ss = self.protein.secondary_structure is not None

            self.seq_target_pssm_ss = np.concatenate([self.protein.secondary_structure, self.seq_target_pssm],
                                                     axis=1) if has_ss else None
            self.seq_target_ss = np.concatenate([self.protein.secondary_structure, self.seq_target],
                                                axis=1) if has_ss else None

        self._metadata['version'] = self._VERSION

        self.target_sequence_length = len(self.protein.sequence)

        modeller_dm = self._get_modeller_dm()

        if modeller_dm is not None:

            if modeller_dm.shape[0] != target_dm.shape[0]:
                LOGGER.info('modeller does not match target for %s' % target)

                modeller_dm = None

            self.modeller_dm, self.modeller_cm = self._gen_cm_dm(modeller_dm)

        self._update_metadata()

    def _gen_cm_dm(self, dm):
        if dm is None:
            return None, None
        nan_inds = np.logical_or(np.isnan(dm), dm == self.NAN_VALUE)
        dm[nan_inds] = self.NAN_VALUE

        cm = (dm < self.THRESHOLD).astype(np.float32)

        cm[nan_inds] = self.NAN_VALUE

        return dm, cm

    def _get_cm_dm_std(self):

        k = len(self.known_structures) - 1
        dms, cms = self._get_k_reference_dm_cm(k)
        if dms is None:
            return
        dms[dms == self.NAN_VALUE] = np.nan
        cms[cms == self.NAN_VALUE] = np.nan

        std_dms = np.nanstd(dms, axis=-1)
        std_dms[np.isnan(std_dms)] = self.NAN_VALUE
        std_cms = np.nanstd(cms, axis=-1)
        std_cms[np.isnan(std_cms)] = self.NAN_VALUE

        return np.concatenate([std_dms, std_cms], axis=-1)

    def _get_angles_outer_product(self, angels_arr):

        phi_psi = np.nan_to_num(np.outer(angels_arr[:, 0], angels_arr[:, 1]),
                                nan=self.NAN_VALUE)
        phi_phi = np.nan_to_num(np.outer(angels_arr[:, 0], angels_arr[:, 0]),
                                nan=self.NAN_VALUE)
        psi_psi = np.nan_to_num(np.outer(angels_arr[:, 1], angels_arr[:, 1]),
                                nan=self.NAN_VALUE)

        arr_stacked = np.stack([psi_psi, phi_phi, phi_psi], axis=2)

        return arr_stacked

    def _get_pssm(self):

        version = min(self._VERSION, self._PSSM_VERSION)

        pssm_file = path.join(self.msa_data_path, 'pssm_v%s.pkl' % version)
        if path.isfile(pssm_file):
            return pkl_load(pssm_file)
        elif self._mode == 'get':
            pssm = self._compute_pssm()
            pkl_save(filename=pssm_file, data=pssm)
            return pssm

    @property
    def sequence_distance_matrix(self):
        def i_minus_j(i, j):
            return np.abs(i - j)

        sequence_distance = np.fromfunction(i_minus_j,
                                            shape=self.target_pdb_cm.shape)
        return sequence_distance

    def _get_metadata(self):
        metadata = {'version': self._VERSION, 'structures': set()}
        metadata_fname = os.path.join(self.msa_data_path, 'metadata.pkl')
        has_metadata = os.path.isfile(metadata_fname)
        if has_metadata:
            tmp_metadata = pkl_load(metadata_fname)
            tmp_metadata['version'] = tmp_metadata.get('version', 0)
            has_known = 'known_structures' in tmp_metadata
            if has_known:
                del tmp_metadata['known_structures']
            tmp_metadata['structures'] = tmp_metadata.get('structures', set())

            metadata = tmp_metadata

        if 'structures' not in metadata:
            raise KeyError('structures not in metadata for %s, metadata: %s' %
                           (self.target, metadata))
        return metadata

    def _update_metadata(self):
        if len(DeepDiff(self._init_m_data, self._metadata)) == 0:
            return
        metadata_fname = os.path.join(self.msa_data_path, 'metadata.pkl')
        pkl_save(metadata_fname, self._metadata)

    @property
    def known_structures(self):
        return set(self._metadata['structures'])

    @property
    def _phylo_structures_mat(self):
        # Returns the phylo genetic distance matrix

        version = self._STRUCTURES_VERSION

        phylo_structures_file = path.join(
            self.msa_data_path, 'structures_phylo_dist_mat_v%s.pkl' % version)
        if os.path.isfile(phylo_structures_file):

            phylo_structures_mat = pd.read_pickle(phylo_structures_file)
            dist_mat_strucs = set(phylo_structures_mat.index)
            has_duplicates = len(
                dist_mat_strucs) != phylo_structures_mat.shape[0]

            if dist_mat_strucs != self.known_structures or has_duplicates:
                os.remove(phylo_structures_file)
                LOGGER.warning(
                    "Phylo dist mat index doesn't match known structures."
                    "\n\nDist mat: %s\n\nKnown: %s" %
                    (phylo_structures_mat.index, self.known_structures))
                return
            return phylo_structures_mat
        elif self._mode == 'get':

            parsed_msa = self._parse_msa()
            self.find_all_structures(parsed_msa)
            # self._generate_phylo_structures_mat(self._parse_msa())

            return pkl_load(phylo_structures_file)

    def _generate_phylo_structures_mat(self, parsed_msa):

        version = min(self._VERSION, self._STRUCTURES_VERSION)

        phylo_structures_file = path.join(
            self.msa_data_path, 'structures_phylo_dist_mat_v%s.pkl' % version)

        known_structures = self._metadata['structures']
        structures_list = list(known_structures)

        msa_structures = [
            str(parsed_msa[s].seq).upper() for s in structures_list
        ]

        id_mat = compute_structures_identity_matrix(msa_structures,
                                                    msa_structures,
                                                    target=str(parsed_msa[self.target].seq.upper()))
        identity_mat = pd.DataFrame(id_mat,
                                    columns=structures_list,
                                    index=structures_list)
        phylo_structures_mat = 1 - identity_mat

        phylo_structures_mat.to_pickle(phylo_structures_file)

    def _get_k_closest_known_structures(self):

        sorted_structures_by_distance = self._get_structues_by_distance()

        if sorted_structures_by_distance is None:
            return

        structures = sorted_structures_by_distance.sort_values(ascending=True)

        return list(structures.index)

    def _get_structues_by_distance(self):
        if self._phylo_structures_mat is None:
            return

        structures_sorted_file = path.join(
            self.msa_data_path, 'structures_sorted.pkl')

        if os.path.isfile(structures_sorted_file):
            return pkl_load(structures_sorted_file)

        struc_dm = self._phylo_structures_mat

        if self.target not in struc_dm or len(self.known_structures) < 2:
            return

        sorted_structures_by_distance = np.clip(struc_dm.loc[:, self.target],
                                                a_min=0.0001,
                                                a_max=1).sort_values(kind='mergesort')

        sorted_structures_by_distance = softmax(
            np.log(1 / sorted_structures_by_distance.loc[
                sorted_structures_by_distance.index != self.target]))

        if len(sorted_structures_by_distance) == 0:
            return

        pkl_save(structures_sorted_file, sorted_structures_by_distance)

        return sorted_structures_by_distance

    @property
    def closest_known_strucutre(self):
        # Returns the closest known structure to the target

        sorted_structures_by_distance = self._get_structues_by_distance()

        if sorted_structures_by_distance is None:
            return

        structure = sorted_structures_by_distance.index[0]

        if structure == self.target:
            if self._mode == 'ask':
                return
            elif self._mode == 'get':

                self.find_all_structures(self._parse_msa())

            raise ValueError('closest structure cannot be the target %s' %
                             self.target)

        return structure

    def _get_sorted_references(self):
        dms = []
        cms = []
        sorted_structures_by_distance = self._get_structues_by_distance()
        with_dist = self._include_local_dist

        for s in sorted_structures_by_distance.index:
            dm, cm = self._get_reference_with_local_dist(
                ref=s) if with_dist else self._get_reference_dm_cm(ref=s)
            dms.append(dm)
            cms.append(cm)

        return dms, cms

    def _get_reference_with_local_dist(self, ref):
        dm, cm = self._get_reference_dm_cm(ref=ref)
        if dm is None:
            # TODO: this needs to be removed
            return None, None
        local_dist = self._get_local_dist_mat(ref)
        if local_dist is None:
            LOGGER.info('No local dist mat for %s' % self.target)
            self.generate_local_seq_dists()
            return None, None
        return np.stack([dm, local_dist], axis=2), np.stack([cm, local_dist],
                                                            axis=2)

    def _get_k_reference_dm_cm(self, k):

        cms = []
        dms = []
        structures = self._get_k_closest_known_structures()

        if structures is None:
            return None, None

        with_dist = self._include_local_dist
        n_structures = len(structures)

        structures = structures if k >= n_structures else structures[
                                                          n_structures - k:n_structures]

        for i in range(min(k, n_structures)):
            s = structures[i]

            dm, cm = self._get_reference_with_local_dist(
                ref=s) if with_dist else self._get_reference_dm_cm(ref=s)
            if dm is None:
                # TODO: this needs to be removed
                return None, None
            cm = cm if len(cm.shape) == 3 else np.expand_dims(cm, 2)
            dm = dm if len(dm.shape) == 3 else np.expand_dims(dm, 2)

            cms.append(cm)
            dms.append(dm)

        if n_structures < k:
            zero_array = np.zeros_like(dms[0])
            masking = [zero_array] * (k - n_structures)
            cms = masking + cms
            dms = masking + dms

        try:
            cms = np.stack(cms, axis=-1)
            dms = np.stack(dms, axis=-1)
        except ValueError:
            LOGGER.info('Problem with structures for %s' % self.target)
            return None, None

        return dms, cms

    def generate_local_seq_dists(self):

        if len(self.known_structures) < 2:
            return

        version = min(self._VERSION, self._MSA_VERSION)
        local_distance = self._LOCAL_DISTANCE
        msa = self._parse_msa()
        target_seq_long = np.array(msa[self.target].seq)
        inds = target_seq_long != '-'
        target_seq = target_seq_long[inds]
        for reference in self.known_structures:

            dm_file = '%s_loc_seq_dist_%s_v%s.pkl' % (reference,
                                                      local_distance, version)

            dm_full_file = os.path.join(self.msa_data_path, dm_file)
            if os.path.isfile(dm_full_file) or reference == self.target:
                continue

            ref_seq = np.array(msa[reference].seq)[inds]
            distance_mat = get_local_sequence_identity_distmat(
                target_seq, ref_seq, local_distance)

            pkl_save(filename=dm_full_file, data=distance_mat)

    def _get_local_dist_mat(self, reference):

        version = min(self._VERSION, self._MSA_VERSION)
        local_distance = self._LOCAL_DISTANCE

        dm_file = '%s_loc_seq_dist_%s_v%s.pkl' % (reference, local_distance,
                                                  version)

        dm_full_file = os.path.join(self.msa_data_path, dm_file)
        if not os.path.isfile(dm_full_file):
            return
        return pkl_load(dm_full_file)

    def _get_reference_dm_cm(self, ref=None):

        version = self._STRUCTURES_VERSION

        reference = self.closest_known_strucutre if ref is None else ref

        if reference is None:
            return None, None

        dm_file = '%s_pdb_v%s.pkl' % (reference, version)

        pdb_dist_mat_file = os.path.join(self.msa_data_path, dm_file)
        if not os.path.isfile(pdb_dist_mat_file):
            return None, None

        dm = pkl_load(pdb_dist_mat_file)
        return self._gen_cm_dm(dm)

    def _get_ec(self):
        # Gets the plmc evolutionary coupling array

        version = min(self._VERSION, self._PLMC_VERSION)

        plmc_file = '%s_v%s.txt' % (self.target, version)
        plmc_path = os.path.join(EVFOLD_PATH, plmc_file)

        if not path.isfile(plmc_path):
            if self._mode == 'ask' or self._mode == 'debug':
                return
            elif self._mode == 'get':
                self._run_plmc()

        pdb_seq = "".join(list(self.protein.sequence))

        def _get_sotred_ec(raw_ec):
            # gets the sorted indices for ec

            raw_ec_local = raw_ec.copy()

            ec_no_na = raw_ec_local.dropna()

            ec_no_dup = ec_no_na.drop_duplicates('i')
            ec_sorted_i = ec_no_dup.sort_values('i')
            ec_sorted_j = ec_no_dup.sort_values('j')

            return ec_sorted_i, ec_sorted_j, ec_no_na

        def _get_msa_seq_ec(raw_ec):
            # returns the msa sequence of the ec data

            raw_ec_local = raw_ec.copy()

            ec_sorted_i, ec_sorted_j, ec = _get_sotred_ec(raw_ec_local)

            # ec shows no duplicates hence i goes until on aa before the last
            last_msa_string = ec_sorted_j.A_j.iloc[-1]

            msa_seq = "".join(list(ec_sorted_i.A_i) + list(last_msa_string))
            return msa_seq

        raw_ec = read_raw_ec_file(plmc_path)
        msa_seq = _get_msa_seq_ec(raw_ec)

        is_valid = msa_seq == pdb_seq

        if not is_valid:
            if self._mode == 'ask':
                return
            elif self._mode == 'get' or self._mode == 'debug':
                self._run_plmc()

                raw_ec = read_raw_ec_file(plmc_path)
                msa_seq = _get_msa_seq_ec(raw_ec)
                is_valid = msa_seq == pdb_seq

                if not is_valid:
                    raise ValueError(
                        'msa sequence must be a sub-string of pdb sequence'
                        '\n\nMSA: %s\n\nPDB: %s\n\n' % (msa_seq, pdb_seq))

        ec_sorted_i, ec_sorted_j, ec = _get_sotred_ec(raw_ec)

        sorted_msa_indices = ec_sorted_i.i.to_list() + ec_sorted_j.j.to_list()
        sorted_msa_indices.sort()

        msa_ind_to_pdb_ind_map = {
            msa: pdb
            for pdb, msa in enumerate(set(sorted_msa_indices))
        }

        pdb_i = ec.loc[:, 'i'].map(msa_ind_to_pdb_ind_map).copy()
        pdb_j = ec.loc[:, 'j'].map(msa_ind_to_pdb_ind_map).copy()

        ec.loc[:, 'pdb_i'] = pdb_i
        ec.loc[:, 'pdb_j'] = pdb_j

        return ec

    def _get_ccmpred(self):

        ccmpred_mat_file = os.path.join(CCMPRED_PATH, f'{self.target}.mat')
        if os.path.isfile(ccmpred_mat_file):
            return np.loadtxt(ccmpred_mat_file)
        elif self._mode == 'get':
            success = self._run_ccmpred()
            if success:
                return np.loadtxt(ccmpred_mat_file)

    def _run_ccmpred(self):

        msa = self._get_no_target_gaps_msa()

        tmp = tempfile.NamedTemporaryFile()

        write_fasta(msa, tmp.name)

        aln_format = os.path.join(HHBLITS_PATH, 'aln',
                                  self.target + '_v%s.aln' % self._MSA_VERSION)

        convert_to_aln(tmp.name, aln_format)
        # reformat = ['reformat.pl', tmp.name, aln_format]
        # subprocess.run(reformat)

        ccmpred_mat_file = os.path.join(CCMPRED_PATH, f'{self.target}.mat')

        ccmpred_cmd = ['ccmpred', aln_format, ccmpred_mat_file]
        p = subprocess.run(ccmpred_cmd)
        if p.returncode != 0:
            LOGGER.info(f'CCMpred failed for {self.target}')
        return p.returncode == 0

    def _run_hhblits(self):
        # Generates multiple sequence alignment using hhblits

        seq = Seq("".join(self.protein.sequence).upper())
        target = self.target
        version = min(self._VERSION, self._MSA_VERSION)

        sequence = SeqIO.SeqRecord(seq, name=target, id=target)
        query = path.join(MSA_DATA_PATH, 'query', target + '.fasta')
        SeqIO.write(sequence, query, "fasta")
        output_hhblits = path.join(HHBLITS_PATH, 'a3m', target + '.a3m')
        output_reformat1 = path.join(HHBLITS_PATH, 'a2m', target + '.a2m')
        output_reformat2 = path.join(HHBLITS_PATH, 'fasta',
                                     target + '_v%s.fasta' % version)

        db_hh = '/cs/zbio/orzuk/projects/ContactMaps/data/MSA_Completion/hh/uniprot20_2016_02/uniprot20_2016_02'

        hhblits = [
            'hhblits', '-i', query, '-d', db_hh, '-n', '3', '-e', '1e-3',
            '-maxfilt', '10000000000', '-neffmax', '20', '-nodiff', '-realign',
            '-realign_max', '10000000000', '-oa3m', output_hhblits
        ]
        subprocess.run(hhblits)
        reformat = ['reformat.pl', output_hhblits, output_reformat1]
        subprocess.run(reformat)

        reformat = ['reformat.pl', output_reformat1, output_reformat2]
        subprocess.run(reformat)

    def _run_plmc(self):
        # Runs 'plmc' to compute evolutionary couplings

        version = min(self._VERSION, self._PLMC_VERSION)

        target_version = (self.target, version)
        file_ec = os.path.join(EVFOLD_PATH, '%s_v%s.txt' % target_version)
        file_params = os.path.join(EVFOLD_PATH, '%s_%s.params' % target_version)
        file_msa = os.path.join(HHBLITS_PATH,
                                'fasta/%s_v%s.fasta' % target_version)

        cmd = 'plmc -o %s -c %s -le 16.0 -m %s -lh 0.01  -g -f %s %s' \
              % (file_params, file_ec, 500, self.target, file_msa)

        subprocess.run(cmd, shell=True)

    def get_plot_reference_data(self, ref):

        parsed_msa = self._parse_msa()

        _, ref_pdb = self._find_structure(
            uniprot=ref,
            msa_sequence=parsed_msa[ref],
            msa_sequence_target=parsed_msa[self.target])

        full_ref_msa = np.array(parsed_msa[ref].seq)
        full_target_msa = np.array(parsed_msa[self.target].seq)

        seq_ref_msa = ''.join(full_ref_msa[full_target_msa != '-'])
        seq_target_msa = ''.join(full_target_msa[full_target_msa != '-'])

        ref_title = f'{ref} ({ref_pdb})'
        target_title = f'Target ({self.target})'
        if len(ref_title) < len(target_title):
            diff = len(target_title) - len(ref_title)
            ref_title += diff * ' '
        elif len(ref_title) > len(target_title):
            diff = len(ref_title) - len(target_title)
            target_title += diff * ' '

        msg = f'{target_title} : {seq_target_msa}\n{ref_title} : {seq_ref_msa}'
        seq_dist = np.round(self._phylo_structures_mat.loc[ref, self.target], 2)
        data = {'msg': msg, 'sequence_distance': float(seq_dist)}

        return data

    def _run_modeller(self, n_structures=1):
        """Runs Modeller to compute pdb file from template which is the closest know structure


        """
        if self.closest_known_strucutre is None:
            return

        version = min(self._VERSION, self._MODELLER_VERSION)

        parsed_msa = self._parse_msa()

        _, reference = self._find_structure(
            uniprot=self.closest_known_strucutre,
            msa_sequence=parsed_msa[self.closest_known_strucutre],
            msa_sequence_target=parsed_msa[self.target])

        if reference is None:
            LOGGER.info('Reference not found for %s' % self.target)
            return

        template_protein, template_chain = reference[0:4], reference[4]

        mod_args = (self._protein_name, self._chain, template_protein,
                    template_chain, n_structures)

        # saving target in pir format
        target_seq = SeqIO.SeqRecord(Seq(''.join(self.protein.sequence), IUPACProtein()),
                                     name='sequence:%s:::::::0.00: 0.00' % self.target, id=self.target, description='')
        # SeqIO.PirIO.PirWriter(open(os.path.join(periscope_path, '%s.ali'%self.target), 'w'))
        SeqIO.write(target_seq, os.path.join(periscope_path, '%s.ali' % self.target), 'pir')

        cmd = f'/cs/staff/dina/modeller9.18/bin/modpy.sh python3 {periscope_path}/run_modeller.py %s %s %s %s %s' % mod_args
        try:
            subprocess.run(cmd, shell=True)
        except urllib.error.URLError:
            raise FileNotFoundError('pdb download error')

        mod_args_2 = (self._protein_name, self._chain, template_protein,
                      template_chain, version, n_structures)

        cmd2 = f'/cs/staff/dina/modeller9.18/bin/modpy.sh python3 {periscope_path}/modeller_files.py %s %s %s %s %s %s' % mod_args_2
        subprocess.run(cmd2, shell=True)

    def get_average_modeller_cm(self, n_structures, version):
        """Returns the average cm over few modeller predicted structures

        Args:
            n_structures (int): number of modeller predicted structures

        Returns:
            np.array: of shape (l, l)

        """

        cms = []

        if self.closest_known_strucutre is None or self.target not in TEST_SET:
            return

        for n_struc in range(1, n_structures + 1):
            pdb_file_path = get_modeller_pdb_file(target=self.target, n_struc=n_struc)
            has_modeller_file = os.path.isfile(pdb_file_path)
            if not has_modeller_file:
                print('shite')
                self._run_modeller(n_structures)
            modeller_prot = Protein(self._protein_name,
                                    self._chain,
                                    pdb_path=MODELLER_PATH,
                                    modeller_n_struc=n_struc,
                                    version=version)
            modeller_dm = modeller_prot.dm
            assert len(modeller_prot.sequence) == len(self.protein.sequence)

            # thres = 8 if version == 4 else 8.5
            #
            cms.append(modeller_dm)

        return np.nanmean(np.stack(cms, axis=2), axis=2)

    def _get_modeller_dm(self):
        """Returns the modeller distance matrix, if exists

        Returns:
            np.array: Modeller's predicted distance matrix

        """

        if self.closest_known_strucutre is None or self.target not in TEST_SET:
            return

        pdb_file_path = get_modeller_pdb_file(target=self.target)
        pdb_file_path_old = get_modeller_pdb_file(target=self.target, version=22)
        if os.path.isfile(pdb_file_path_old):
            LOGGER.info(f'Renaming {pdb_file_path_old} to {pdb_file_path}')
            os.rename(pdb_file_path_old, pdb_file_path)

        has_modeller_file = os.path.isfile(pdb_file_path)

        if not has_modeller_file and self._mode == 'ask':
            LOGGER.warning('Modeller pdb file %s not found' % pdb_file_path)
            return

        if self._mode == 'get' and not has_modeller_file:
            self._run_modeller()

        modeller_prot = Protein(self._protein_name,
                                self._chain,
                                pdb_path=MODELLER_PATH,
                                version=self._STRUCTURES_VERSION)

        def _fix_dm_mismtach(modeller_protein, pdb_protein):
            modeller_seq = modeller_protein.sequence
            pdb_seq = pdb_protein.sequence
            if len(modeller_seq) != len(pdb_seq) - 1:
                return

            pdb_seq_truncated = pdb_seq[0:len(modeller_seq)]

            is_last = ''.join(modeller_seq) == ''.join(pdb_seq_truncated)

            mismatch_ind = len(pdb_seq) if is_last else np.min(
                np.where(modeller_seq != pdb_seq_truncated))

            pdb_seq_aligned = np.concatenate(
                [pdb_seq[0:mismatch_ind], pdb_seq[mismatch_ind + 1:]], axis=0)

            if "".join(pdb_seq_aligned) != ''.join(modeller_seq):
                return

            modeller_dm = np.ones_like(pdb_protein.dm) * -1

            modeller_dm[0:mismatch_ind,
            0:mismatch_ind] = modeller_protein.dm[0:mismatch_ind,
                              0:mismatch_ind]
            modeller_dm[mismatch_ind + 1:,
            mismatch_ind + 1:] = modeller_protein.dm[mismatch_ind:,
                                 mismatch_ind:]

            return modeller_dm

        if not hasattr(modeller_prot, 'dm'):
            return

        modeller_dm = modeller_prot.dm
        if len(modeller_prot.sequence) == len(self.protein.sequence) - 1:
            LOGGER.info(f'Modeller mismatch for {self.target}')

            modeller_dm = _fix_dm_mismtach(modeller_prot, self.protein)
            if modeller_dm is None:
                return

        try:
            diff = np.abs(modeller_dm - self.protein.dm).mean()
        except ValueError:
            LOGGER.info('Sequence mismatch\nM: %s\nP: %s\n' % (''.join(
                modeller_prot.sequence), "".join(self.protein.sequence)))

        return modeller_dm

    def _get_no_target_gaps_msa(self):
        version = min(self._VERSION, self._MSA_VERSION)

        msa_file = os.path.join(HHBLITS_PATH, 'fasta',
                                self.target + '_v%s.fasta' % version)

        full_alphabet = Gapped(ExtendedIUPACProtein())
        fasta_seqs = [f.upper() for f in list(SeqIO.parse(msa_file, "fasta", alphabet=full_alphabet))]
        target_seq_full = fasta_seqs[0].seq
        target_seq_no_gap_inds = [i for i in range(len(target_seq_full)) if target_seq_full[i] != '-']
        target_seq = ''.join(target_seq_full[i] for i in target_seq_no_gap_inds)
        def _slice_seq(seq, inds):
            seq.seq = Seq(''.join(seq.seq[i] for i in inds), alphabet=full_alphabet)
            return seq

        assert target_seq == "".join(self.protein.sequence)
        fasta_seqs_short = [_slice_seq(s, target_seq_no_gap_inds) for s in fasta_seqs]
        return fasta_seqs_short

    def _get_bio_pssm(self):
        """Pssm array

        Returns:
            np.array: the msa pssm of shape (l, 26)

        """

        version = min(self._VERSION, self._MSA_VERSION)

        pssm_file = path.join(self.msa_data_path, 'pssm_bio_v%s.pkl' % version)

        if os.path.isfile(pssm_file):
            return pkl_load(pssm_file)

        msa_file = os.path.join(HHBLITS_PATH, 'fasta',
                                self.target + '_v%s.fasta' % version)

        full_alphabet = Gapped(ExtendedIUPACProtein())

        fasta_seqs = [f.upper() for f in list(SeqIO.parse(msa_file, "fasta", alphabet=full_alphabet))]
        target_seq_full = fasta_seqs[0].seq
        target_seq_no_gap_inds = [i for i in range(len(target_seq_full)) if target_seq_full[i] != '-']
        target_seq = ''.join(target_seq_full[i] for i in target_seq_no_gap_inds)

        def _slice_seq(seq, inds):
            seq.seq = Seq(''.join(seq.seq[i] for i in inds), alphabet=full_alphabet)
            return seq

        assert target_seq == "".join(self.protein.sequence)
        fasta_seqs_short = [_slice_seq(s, target_seq_no_gap_inds) for s in fasta_seqs]
        msa = MultipleSeqAlignment(fasta_seqs_short)
        summary = AlignInfo.SummaryInfo(msa)

        short_alphabet = IUPACProtein()
        chars_to_ignore = list(set(full_alphabet.letters).difference(set(short_alphabet.letters)))

        n_homologous = len(msa)

        expected = collections.OrderedDict(sorted(AMINO_ACID_STATS.items()))
        expected = {k: n_homologous * v / 100 for k, v in expected.items()}
        pssm = summary.pos_specific_score_matrix(chars_to_ignore=chars_to_ignore)
        pssm = np.array([list(p.values()) for p in pssm]) / np.array(list(expected.values()))[:, None].T

        epsilon = 1e-06

        pssm_log = -1 * np.log(np.clip(pssm, a_min=epsilon, a_max=None))

        pkl_save(pssm_file, pssm_log)

        return pssm

    def _parse_msa(self):
        """Parses the msa data

        Returns:
            dict[str,SeqRecord]: homologos name to sequence object mapping

        """

        version = min(self._VERSION, self._MSA_VERSION)

        parsed_msa_file = path.join(self.msa_data_path,
                                    'parsed_msa_v%s.pkl' % version)

        if os.path.isfile(parsed_msa_file):
            return pkl_load(parsed_msa_file)

        msa_file = os.path.join(HHBLITS_PATH, 'fasta',
                                self.target + '_v%s.fasta' % version)

        if not os.path.isfile(msa_file):
            if self._mode == 'get':
                LOGGER.info('Running HHblits for %s' % self.target)
                self._run_hhblits()
            else:
                LOGGER.info('MSA for %s not found' % self.target)
                return

        def _get_id(seq):
            if seq.id == self.target:
                return seq.id
            return seq.id.split('|')[1]

        fasta_seqs = list(SeqIO.parse(msa_file, "fasta", alphabet=Gapped(ExtendedIUPACProtein())))

        sequences = {s.id: s for s in list(SeqIO.parse(msa_file, "fasta", alphabet=Gapped(ExtendedIUPACProtein())))}
        # sequences = {
        #     _get_id(seq): SeqRecord(seq.seq.upper(),
        #                             id=seq.id)
        #     for seq in fasta_seqs
        # }

        return sequences

    def _get_numeric_msa(self, parsed_msa, target_seq, n=None):

        target_sequence_msa = ''.join(target_seq)
        pdb_sequence = "".join(self.protein.sequence)

        vals = parsed_msa.values()
        if n is not None:
            keys = np.random.choice(list(parsed_msa.keys()),
                                    size=n,
                                    replace=True)
            vals = [parsed_msa[k] for k in keys]

        homologos = pd.DataFrame(vals).values
        alignment_pdb = np.zeros(shape=(len(homologos), len(pdb_sequence)),
                                 dtype='<U1')
        alignment_pdb[:] = '-'

        pdb_inds, msa_inds = self._align_pdb_msa(
            pdb_sequence, target_sequence_msa, list(range(len(pdb_sequence))))

        alignment_pdb[:, pdb_inds] = homologos[:, msa_inds]

        aa = list('-ACDEFGHIKLMNPQRSTVWYX')
        aa_dict = {aa[i]: i for i in range(len(aa))}
        aa_dict['Z'] = 0
        aa_dict['B'] = 0
        aa_dict['U'] = 0
        aa_dict['O'] = 0

        numeric_msa = np.vectorize(aa_dict.__getitem__)(alignment_pdb).astype(
            np.int32)

        return numeric_msa

    def _compute_pssm(self):
        """Computes 2-d pssm array

        Args:
            parsed_msa (dict[str,SeqRecord]): Msa data

        Returns:
            np.array: of shape (l,l,44)

        """
        msa = self._parse_msa()
        numeric_msa = self._get_numeric_msa(msa,
                                            target_seq=msa[self.target].seq)

        pssm = compute_pssm(numeric_msa)
        return pssm

    def _one_hot_msa(self, numeric_msa):

        msa = np.expand_dims(numeric_msa, 2)

        def _bow_prot(a, axis):
            msa_bow = []

            for hom in a:
                bow = []

                for aa in list(hom):
                    aa_numeric = np.zeros(PROTEIN_BOW_DIM)
                    aa_numeric[int(aa)] = 1.0
                    bow.append(aa_numeric)
                bow_arr = np.array(bow)
                msa_bow.append(bow_arr)
            return np.array(msa_bow)

        bow_msa = np.apply_over_axes(func=_bow_prot, a=msa, axes=[0])

        return bow_msa

    def _get_seq_refs_full(self):
        """Numeric sequence representation of references

        Returns:
            np.array: of shape (l, PROTEIN_BOW_DIM, n_structures)

        """

        seq_refs_file = path.join(self.msa_data_path, 'seq_refs_full_2.pkl')

        if os.path.isfile(seq_refs_file):
            return pkl_load(seq_refs_file)

        refs = self._get_k_closest_known_structures()

        if refs is None:
            return

        bow_msa_refs = self.bow_msa(refs=refs, n=None).transpose(1, 2, 0)

        pkl_save(filename=seq_refs_file, data=bow_msa_refs)

        return bow_msa_refs

    def _get_seq_refs(self, k):
        """Numeric sequence representation of references

        Returns:
            np.array: of shape (l, PROTEIN_BOW_DIM, k)

        """

        bow_msa_full = self._get_seq_refs_full()

        if bow_msa_full is None:
            return

        shape = bow_msa_full.shape
        n_refs = shape[-1]

        if n_refs >= k:
            return bow_msa_full[..., 0:k]
        else:
            return np.concatenate(
                [np.zeros((shape[0], shape[1], k - n_refs)), bow_msa_full],
                axis=2)

    def _get_ss_refs(self, k):
        """Numeric sequence representation of references

        Returns:
            np.array: of shape (l, PROTEIN_BOW_DIM, k)

        """

        ss_refs_full = self._get_aligned_ss()

        if ss_refs_full is None:
            return

        shape = ss_refs_full.shape
        n_refs = shape[-1]

        if n_refs >= k:
            return ss_refs_full[..., 0:k]
        else:
            return np.concatenate(
                [np.zeros((shape[0], shape[1], k - n_refs)), ss_refs_full],
                axis=2)

    def _get_seq_target(self):
        """Numeric sequence representation of target

        Returns:
            np.array: of shape (l, PROTEIN_BOW_DIM)

        """

        seq_target_file = path.join(self.msa_data_path, 'seq_target.pkl')

        if os.path.isfile(seq_target_file):
            return np.squeeze(pkl_load(seq_target_file))

        bow_msa_target = np.squeeze(self.bow_msa(refs=[self.target], n=None))

        pkl_save(filename=seq_target_file, data=bow_msa_target)

        return bow_msa_target

    def _get_aligned_ss(self):

        ss_refs_file = os.path.join(self.msa_data_path, 'ss_refs.pkl')
        if os.path.isfile(ss_refs_file):
            return pkl_load(ss_refs_file)

        refs = self._get_k_closest_known_structures()
        if refs is None:
            return

        parsed_msa = self._parse_msa()
        target_seq_msa = parsed_msa[self.target].seq
        valid_inds = [i for i in range(len(target_seq_msa)) if target_seq_msa[i] != '-']
        parsed_msa = {r: parsed_msa[r].seq for r in refs}
        na_arr = np.array([0, 0, 0, 0, 0, 0, 0, 1])
        secondary_structures = {r: np.stack([na_arr] * len(target_seq_msa), axis=0) for r in
                                refs}
        for ref, seq in parsed_msa.items():
            mapping = self._get_map_wrapper(homologous=ref, msa_sequence=str(seq), one_d=True)
            ss = mapping['protein'].secondary_structure
            secondary_structures[ref][mapping['msa'], :] = ss[mapping['pdb'], :]
            secondary_structures[ref] = secondary_structures[ref][valid_inds, :]

        aligned_ss = np.stack([secondary_structures[s] for s in refs], axis=2)
        pkl_save(ss_refs_file, aligned_ss)
        return aligned_ss

    def bow_msa(self, refs=None, n=NUM_HOMOLOGOUS):
        """

        Args:
            n (int): number of homologous
            refs (list[str]): references to use as sub-alignment

        Returns:
            np.array: of shape (n, l, 22)

        """
        parsed_msa = self._parse_msa()
        target_seq = parsed_msa[self.target].seq

        if parsed_msa is None:
            return

        if refs is not None:
            parsed_msa = {r: parsed_msa[r] for r in refs}

        numeric_msa = self._get_numeric_msa(parsed_msa, target_seq, n)

        return self._one_hot_msa(numeric_msa)

    def _get_plmc_score(self):
        # return the plmc score matrix of shape (l, l)

        if self._ec is None:
            return

        target_indices_pdb = (np.array(self._ec.pdb_i).astype(np.int64),
                              np.array(self._ec.pdb_j).astype(np.int64))

        l = len(self.protein.sequence)

        plmc_score = np.zeros(shape=(l, l))

        plmc_score[target_indices_pdb] = self._ec['cn'].values

        return plmc_score

    @staticmethod
    def _align_pdb_msa(pdb_sequence, msa_sequence, pdb_indices, one_d=False):
        """Get alined indices if exists

        Args:
            pdb_sequence (str): Amino acid sequence
            msa_sequence (str): Amino acid and gaps sequence
            pdb_indices (list[int]): Pdb indices according to sifts mapping
            one_d (bool): if true we return 1 dimentional

        Returns:
            tuple[list[int, int], list[int, int]]: matching x-y indices for pdb and msa arrays

        """

        msa_sequence_no_gaps = msa_sequence.replace('-', "")
        pdb_msa_substring = msa_sequence_no_gaps.find(pdb_sequence)
        msa_pdb_substring = pdb_sequence.find(msa_sequence_no_gaps)

        msa_indices = [
            i for i in range(len(msa_sequence)) if msa_sequence[i] != '-'
        ]

        if pdb_msa_substring == -1 and msa_pdb_substring == -1:
            # LOGGER.info(
            #     "Could not align sequences:\n\nMSA: %s\n\nPDB: %s\n\n" %
            #     (msa_sequence_no_gaps, pdb_sequence))
            return None, None
        elif pdb_msa_substring != -1:

            start_msa = pdb_msa_substring
            end_msa = start_msa + len(pdb_sequence)
            msa_indices = msa_indices[start_msa:end_msa]

        elif msa_pdb_substring != -1:
            start_pdb = msa_pdb_substring
            end_pdb = msa_pdb_substring + len(msa_sequence_no_gaps)
            pdb_indices = pdb_indices[start_pdb:end_pdb]

        if one_d:
            return pdb_indices, msa_indices

        pairs_pdb = list(itertools.combinations(pdb_indices, 2))
        pairs_msa = list(itertools.combinations(msa_indices, 2))

        pdb_inds, msa_inds = list(zip(*pairs_pdb)), list(zip(*pairs_msa))

        if len(pdb_inds) == 0:
            return None, None

        return pdb_inds, msa_inds

    def _get_map_wrapper(self, homologous, msa_sequence, one_d):
        is_target = homologous == self.target
        selected_structures = SIFTS_MAPPING.get(homologous, None)
        if is_target:
            selected_structures = [[
                homologous, {
                    'pdb': (1, len(self.protein.sequence) + 1)
                }
            ]]
        if selected_structures is None:
            return None, None
        for protein_map in selected_structures:

            protein = protein_map[0][0:4]
            chain = protein_map[0][4]

            if protein == self._protein_name and not is_target:
                continue
            try:
                reference_protein = Protein(protein, chain, version=self._STRUCTURES_VERSION)

            except (FileNotFoundError, IndexError):
                continue

            pdb_sequence_full = ''.join(reference_protein.sequence)

            reference_protein_length = len(pdb_sequence_full)

            start_ind = min(protein_map[1]['pdb'][0] - 1,
                            reference_protein_length)
            end_ind = min(protein_map[1]['pdb'][1] - 1,
                          reference_protein_length)

            pdb_indices = list(range(start_ind, end_ind))

            pdb_sequence = pdb_sequence_full[start_ind:end_ind]

            pdb_inds, msa_inds = self._align_pdb_msa(pdb_sequence,
                                                     msa_sequence,
                                                     pdb_indices,
                                                     one_d)

            return {'pdb': pdb_inds, 'msa': msa_inds, 'protein': reference_protein}

    def _find_structure(self, uniprot, msa_sequence, msa_sequence_target):
        """

        Args:
            uniprot (str): Uniprot protein name
            msa_sequence (SeqRecord): Row of the msa
            msa_sequence_target (SeqRecord): Target row of the msa

        Returns:
            tuple[np.array, str]: aligned dist mat of shape l * l and protein pdb name

        """

        msa_sequence = str(msa_sequence.seq)
        msa_sequence_target = str(msa_sequence_target.seq)

        is_target = uniprot == self.target
        has_mapping = uniprot in SIFTS_MAPPING

        if not is_target and not has_mapping:
            return None, None

        if is_target:
            selected_structures = [[
                uniprot, {
                    'pdb': (1, len(self.protein.sequence) + 1)
                }
            ]]
        else:
            selected_structures = SIFTS_MAPPING[uniprot]

        for protein_map in selected_structures:

            protein = protein_map[0][0:4]
            chain = protein_map[0][4]

            if protein == self._protein_name and not is_target:
                continue
            try:
                reference_protein = Protein(protein, chain, version=self._STRUCTURES_VERSION)

            except (FileNotFoundError, IndexError):
                continue

            pdb_sequence_full = ''.join(reference_protein.sequence)

            reference_protein_length = len(pdb_sequence_full)

            start_ind = min(protein_map[1]['pdb'][0] - 1,
                            reference_protein_length)
            end_ind = min(protein_map[1]['pdb'][1] - 1,
                          reference_protein_length)

            pdb_indices = list(range(start_ind, end_ind))

            pdb_sequence = pdb_sequence_full[start_ind:end_ind]

            pdb_inds, msa_inds = self._align_pdb_msa(pdb_sequence,
                                                     msa_sequence,
                                                     pdb_indices)

            if pdb_inds is None:
                return None, None

            if is_target:
                return self.protein.dm, self.target

            m = len(msa_sequence)

            sequence_length = len(self.protein.sequence)

            msa_mat = np.ones(shape=(m, m)) * np.nan
            aligned_mat = np.ones(shape=(sequence_length,
                                         sequence_length)) * np.nan

            aligned_distance_mat_values = reference_protein.dm[pdb_inds[0],
                                                               pdb_inds[1]]

            msa_mat[msa_inds[0], msa_inds[1]] = aligned_distance_mat_values

            target_sequence = ''.join(self.protein.sequence)
            target_pdb_indices = list(range(0, len(target_sequence)))

            pdb_inds_target, msa_inds_target = self._align_pdb_msa(
                target_sequence, msa_sequence_target, target_pdb_indices)

            aligned_mat[pdb_inds_target[0],
                        pdb_inds_target[1]] = msa_mat[msa_inds_target[0],
                                                      msa_inds_target[1]]
            aligned_mat[pdb_inds_target[1],
                        pdb_inds_target[0]] = msa_mat[msa_inds_target[0],
                                                      msa_inds_target[1]]

            np.fill_diagonal(aligned_mat, 0)

            return np.nan_to_num(aligned_mat,
                                 nan=self.NAN_VALUE), protein + chain

        return None, None

    def _find_structure_permissive(self, uniprot, msa_sequence, msa_sequence_target):
        """

        Args:
            uniprot (str): Uniprot protein name
            msa_sequence (SeqRecord): Row of the msa
            msa_sequence_target (SeqRecord): Target row of the msa

        Returns:
            tuple[np.array, str]: aligned dist mat of shape l * l and protein pdb name

        """

        msa_sequence = str(msa_sequence.seq)
        msa_sequence_target = str(msa_sequence_target.seq)

        is_target = uniprot == self.target
        has_mapping = uniprot in SIFTS_MAPPING

        if not is_target and not has_mapping:
            return None, None

        if is_target:
            selected_structures = [[
                uniprot, {
                    'pdb': (1, len(self.protein.sequence) + 1)
                }
            ]]
        else:
            selected_structures = SIFTS_MAPPING[uniprot]

        for protein_map in selected_structures:

            protein = protein_map[0][0:4]
            chain = protein_map[0][4]

            if protein == self._protein_name and not is_target:
                continue
            try:
                reference_protein = Protein(protein, chain, version=self._STRUCTURES_VERSION)

            except (FileNotFoundError, IndexError):
                continue

            # pdb_sequence_full = ''.join(reference_protein.sequence)

            uniprot_seq = str(_get_uniprot_seq(uniprot).seq)
            uniprot_seq_length = len(uniprot_seq)

            start_ind = min(protein_map[1]['uniprot'][0] - 1,
                            uniprot_seq_length)
            end_ind = min(protein_map[1]['uniprot'][1] - 1,
                          uniprot_seq_length)

            # reference_protein_length = len(pdb_sequence_full)
            #
            # start_ind = min(protein_map[1]['pdb'][0] - 1,
            #                 reference_protein_length)
            # end_ind = min(protein_map[1]['pdb'][1] - 1,
            #               reference_protein_length)

            pdb_indices = list(range(start_ind, end_ind))

            pdb_sequence = uniprot_seq[start_ind:end_ind]

            pdb_inds, msa_inds = self._align_pdb_msa(pdb_sequence,
                                                     msa_sequence,
                                                     pdb_indices)

            if pdb_inds is None:
                return None, None

            if is_target:
                return self.protein.dm, self.target

            m = len(msa_sequence)

            sequence_length = len(self.protein.sequence)

            msa_mat = np.ones(shape=(m, m)) * np.nan
            aligned_mat = np.ones(shape=(sequence_length,
                                         sequence_length)) * np.nan

            aligned_distance_mat_values = reference_protein.dm[pdb_inds[0],
                                                               pdb_inds[1]]

            msa_mat[msa_inds[0], msa_inds[1]] = aligned_distance_mat_values

            target_sequence = ''.join(self.protein.sequence)
            target_pdb_indices = list(range(0, len(target_sequence)))

            pdb_inds_target, msa_inds_target = self._align_pdb_msa(
                target_sequence, msa_sequence_target, target_pdb_indices)

            aligned_mat[pdb_inds_target[0],
                        pdb_inds_target[1]] = msa_mat[msa_inds_target[0],
                                                      msa_inds_target[1]]
            aligned_mat[pdb_inds_target[1],
                        pdb_inds_target[0]] = msa_mat[msa_inds_target[0],
                                                      msa_inds_target[1]]

            np.fill_diagonal(aligned_mat, 0)

            return np.nan_to_num(aligned_mat,
                                 nan=self.NAN_VALUE), protein + chain

        return None, None

    def _has_structure_data(self):
        version = self._STRUCTURES_VERSION
        known_structures = self.known_structures
        for homologous in known_structures:

            structure_file = os.path.join(
                self.msa_data_path, '%s_pdb_v%s.pkl' % (homologous, version))
            if not os.path.exists(structure_file):
                return False
        return True

    def _gen_structures_version(self):

        LOGGER.info(f'updating structures for {self.target} to version : {self._STRUCTURES_VERSION}')

        parsed_msa = self._parse_msa()

        version = self._STRUCTURES_VERSION

        known_structures = self.known_structures
        LOGGER.info('Known structures are %s' % known_structures)
        target_msa_seq = parsed_msa[self.target]

        unaligned_structures = set()

        for homologous in known_structures:

            structure_file = os.path.join(
                self.msa_data_path, '%s_pdb_v%s.pkl' % (homologous, version))
            if os.path.exists(structure_file):
                continue
            aligned_structure, _ = self._find_structure(
                uniprot=homologous,
                msa_sequence=parsed_msa[homologous],
                msa_sequence_target=target_msa_seq)

            if aligned_structure is None:
                LOGGER.warning('aligned structure not found for %s in msa %s' %
                               (homologous, self.target))
                unaligned_structures.add(homologous)

            pkl_save(filename=structure_file, data=aligned_structure)

        self._metadata['structures_date'] = datetime.date.today().strftime(
            "%B_%d")

        self._metadata['version'] = version
        self._metadata['structures'] = known_structures.difference(
            unaligned_structures)

        old_structures = [
            f for f in os.listdir(self.msa_data_path)
            if f.endswith('pdb_v21.pkl')
        ]
        for s in old_structures:
            if not os.path.isfile(s):
                continue
            os.remove(os.path.join(self.msa_data_path, s))

        self._generate_phylo_structures_mat(parsed_msa)

        self._update_metadata()

    def find_all_structures(self, parsed_msa):
        # finds and saves all known strctures in an alignment

        LOGGER.info('finding structures for msa %s' % self.target)

        version = self._STRUCTURES_VERSION

        known_structures = set()

        target_msa_seq = parsed_msa[self.target]

        for homologous in parsed_msa:

            aligned_structure, _ = self._find_structure(
                uniprot=homologous,
                msa_sequence=parsed_msa[homologous],
                msa_sequence_target=target_msa_seq)

            if aligned_structure is None:
                continue

            known_structures.add(homologous)

            structure_file = os.path.join(
                self.msa_data_path, '%s_pdb_v%s.pkl' % (homologous, version))

            pkl_save(filename=structure_file, data=aligned_structure)

        self._metadata['structures'] = known_structures
        self._metadata['structures_date'] = datetime.date.today().strftime(
            "%B_%d")

        self._generate_phylo_structures_mat(parsed_msa)

        self._update_metadata()
