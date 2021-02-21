import logging
import os
import shutil
import subprocess
import warnings

import Bio
from Bio import SeqIO, PDB
from Bio.Data import SCOPData
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
from Bio.PDB.StructureBuilder import PDBConstructionWarning

import numpy as np
from Bio.PDB import Polypeptide, is_aa
from Bio.PDB.PDBParser import PDBParser
from .constants import PATHS
from .utils import pkl_load, get_modeller_pdb_file, get_pdb_fname, check_path, pkl_save

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
warnings.simplefilter('ignore', PDBConstructionWarning)
path_to_dssp = os.path.join(PATHS.src, 'dssp-2.3.0/mkdssp')


def _is_valid(r):
    return is_aa(r)


class Protein:
    NAN_VALUE = -1
    _THRESHOLD = 8

    def __init__(self, protein, chain, pdb_path=PATHS.pdb, modeller_n_struc=1, version=6, pdb_fname=None):
        self.VERSION = version
        self.protein = protein if type(protein) == str else protein.decode(
            "utf-8")
        self.chain = chain.upper() if type(chain) == str else chain.decode("utf-8").upper()
        self.pdb_parser = PDBParser(PERMISSIVE=1)

        target = self.protein + self.chain
        self.target = target
        self._target_dir = os.path.join(PATHS.proteins, protein[1:3], target)
        check_path(self._target_dir)

        # self.pdb_path = pdb_path if type(pdb_path) == str else pdb_path.decode(
        #     "utf-8")

        self._is_modeller = pdb_path == PATHS.modeller
        self.pdb_fname = get_pdb_fname(self.protein)
        if self._is_modeller:
            self.pdb_fname = get_modeller_pdb_file(target, n_struc=modeller_n_struc, templates=True)

        if pdb_fname is not None:
            self.pdb_fname = pdb_fname

        if not os.path.exists(self.pdb_fname):
            if self._is_modeller:
                LOGGER.info('no modeller file found %s' % self.pdb_fname)
                return
            success_code = self.get_pdb_file()
            if success_code != 0:
                return

    @property
    def str_seq(self):
        try:
            return "".join(self.sequence)
        except Exception:
            return

    def get_pdb_file(self):
        if self.protein.upper()== self.protein:
            return

        msg = "cannot find a pdb file named '" + self.pdb_fname + "'."
        LOGGER.info(msg)

        pdb_file_zipped = 'pdb%s.ent.gz' % self.protein

        ftp_file = 'ftp://ftp.wwpdb.org/pub/pdb/data/structures/divided/pdb/%s/%s' % (
            self.protein[1:3], pdb_file_zipped)

        ftp_cmd = "wget %s -P %s -q" % (ftp_file, PATHS.pdb)

        subprocess.Popen(ftp_cmd, shell=True, stdout=open(os.devnull, 'wb')).wait()
        cmd2 = 'gunzip -f ' + os.path.join(PATHS.pdb, pdb_file_zipped)
        if os.path.isfile(os.path.join(PATHS.pdb, pdb_file_zipped)):
            success = subprocess.Popen(cmd2, shell=True).wait()
        else:
            success = 1

        if success != 0:
            return

        check_path(os.path.join(PATHS.data, 'pdb', self.protein[1:3]))

        shutil.move(os.path.join(PATHS.pdb, f'pdb{self.protein}.ent'), get_pdb_fname(self.protein))

        return success

    @property
    def sequence(self):
        return self.get_chain_seq()

    @property
    def model(self):
        structure = self.pdb_parser.get_structure(id='strcuture',
                                                  file=self.pdb_fname)
        model = structure[0]
        return model

    @property
    def _bio_chain(self):
        try:
            chain = self.model[self.chain] if not self._is_modeller else list(self.model.get_chains())[0]
        except KeyError:
            return
        return chain

    @property
    def _aa_mask(self):

        poly = Polypeptide.Polypeptide(self._bio_chain)
        aa_mask = [Polypeptide.is_aa(r) for r in poly]
        return aa_mask

    def _get_residues(self):

        residues = [r for r in self._bio_chain.get_residues() if _is_valid(r)]

        def _res_number(r):
            return r.id[1]

        residues = sorted(residues, key=_res_number)

        return residues

    def _get_residues_hetatm(self):

        residues = [r for r in self._bio_chain.get_residues() if r.full_id[3][0] != 'W']
        return residues

    def get_dssp_dict(self):
        dssp_dict = dssp_dict_from_pdb_file(self.pdb_fname)[0]
        return dssp_dict

    @property
    def modeller_str_seq_hetatm(self):

        residues = self._get_residues_hetatm()

        return "".join(self._get_one_letter(r) for r in residues)

    @staticmethod
    def _get_one_letter(res):
        if isinstance(res, PDB.Residue.DisorderedResidue):
            return SCOPData.protein_letters_3to1.get(res.disordered_get_id_list()[0], ".")
        return SCOPData.protein_letters_3to1.get(res.get_resname(), ".")

    @property
    def modeller_str_seq(self):

        residues = self._get_residues()

        return "".join(self._get_one_letter(r) for r in residues)

    @property
    def modeller_start_end(self):
        residues = self._get_residues()

        return residues[0].full_id[3][1], residues[-1].full_id[3][1]

    @property
    def dssp(self):
        dssp = Bio.PDB.DSSP(self.model,
                            self.pdb_fname,
                            dssp=path_to_dssp)
        return dssp

    @property
    def ss_acc(self):
        ss = self._get_secondary_structure()
        acc = self._get_solvent_accessibility()
        return np.concatenate([ss, acc], axis=1)

    def _get_solvent_accessibility(self):
        na_arr = np.empty(shape=(len(self.str_seq), 1))
        na_arr[:] = np.nan
        try:
            dssp = self.dssp
        except Exception:
            return na_arr

        accessible_surface_area = []

        for residue in self._get_residues():
            if _is_valid(residue):
                try:

                    accessible_surface_area.append(
                        dssp[residue.get_full_id()[2:]][3])
                    if accessible_surface_area[-1] == 'NA':
                        accessible_surface_area[-1] = np.nan
                except:
                    accessible_surface_area.append(np.nan)

        accessible_surface_area = np.expand_dims(np.array(
            accessible_surface_area, dtype=np.float32), axis=1)

        return accessible_surface_area

    def _get_secondary_structure(self):

        ss_file = os.path.join(
            self._target_dir, 'secondary_structure_v%s.pkl' % self.VERSION)

        if os.path.isfile(ss_file):
            ss = pkl_load(ss_file)
            if ss.shape[0] == len(self.str_seq):
                return ss

        one_hot_dict = {}
        ss_keys = ['H', 'B', 'E', 'G', 'I', 'T', 'S', '-']
        for i, k in enumerate(ss_keys):
            one_hot_dict[k] = np.zeros(shape=(len(ss_keys),))
            one_hot_dict[k][i] = 1

        ss_nas = np.stack([one_hot_dict['-'] for i in range(len(self.str_seq))])

        try:
            dssp_dict = self.get_dssp_dict()
        except Exception:
            return ss_nas

        dssp_seq_arr = np.array([dssp_dict[k][0] for k in dssp_dict if k[0] == self.chain])

        # inds = self._aa_mask[0:len(dssp_seq_arr)]

        dssp_seq = "".join(dssp_seq_arr)

        if len(self.str_seq) != len(dssp_seq):
            return ss_nas
        ss = np.stack([one_hot_dict[dssp_dict[k][1]] for k in dssp_dict if k[0] == self.chain])

        pkl_save(ss_file, ss)
        return ss

    def get_chain_seq(self):

        seq_file = os.path.join(self._target_dir, 'seq_aa.pkl')
        if os.path.isfile(seq_file):
            return pkl_load(seq_file)
        residues = self._get_residues()

        seq = np.array([self._get_one_letter(r) for r in residues])
        pkl_save(seq_file, seq)
        return seq

        # target_seq_file = os.path.join(
        #     self._target_dir, 'target_sequence_v%s.pkl' % self.VERSION)
        # if os.path.isfile(target_seq_file) and not self._is_modeller:
        #     return pkl_load(target_seq_file)
        #
        # poly = Polypeptide.Polypeptide(self._bio_chain)
        # sequence = np.array(poly.get_sequence())
        # try:
        #     clean_sequence = sequence[self._aa_mask]
        # except IndexError:
        #     return
        # # if not self._is_modeller:
        # #     pkl_save(target_seq_file, clean_sequence)
        # return clean_sequence

    def get_chain_seq_full(self):

        # target_seq_file = os.path.join(
        #     self._target_dir, 'target_sequence_v%s.pkl' % self.VERSION)
        # if os.path.isfile(target_seq_file) and not self._is_modeller:
        #     return pkl_load(target_seq_file)

        protein, chain_id = self.protein, self.chain
        pdb_file = self.pdb_fname
        chain = {record.id: record.seq for record in SeqIO.parse(pdb_file, 'pdb-seqres')}
        query_chain_id = chain_id if chain_id in chain else f'{protein.upper()}:{chain_id}'
        clean_sequence = np.array([aa for aa in chain[query_chain_id]])
        #
        # if not self._is_modeller:
        #     pkl_save(target_seq_file, clean_sequence)
        return clean_sequence

    def get_torsion_angle(self):

        target_file = os.path.join(self._target_dir,
                                   'target_angels_v%s.pkl' % self.VERSION)
        if os.path.isfile(target_file) and not self._is_modeller:
            return pkl_load(target_file)

        poly = Polypeptide.Polypeptide(self._bio_chain)
        all_angles = np.array(poly.get_phi_psi_list())
        angels_arr = np.array(all_angles[self._aa_mask], dtype=np.float32)
        # if not self._is_modeller:
        #     pkl_save(target_file, angels_arr)
        return angels_arr

    def _calc_residue_dist(self, residue_one, residue_two, dist_atoms='CA'):
        """Returns the C-alpha distance between two residues"""

        if not Polypeptide.is_aa(residue_one) or not Polypeptide.is_aa(residue_two):
            return np.nan
        dist_atom_1 = dist_atoms if dist_atoms in residue_one else 'CA'
        dist_atom_2 = dist_atoms if dist_atoms in residue_two else 'CA'

        try:
            diff_vector = residue_one[dist_atom_1].coord - residue_two[dist_atom_2].coord
        except KeyError:
            return np.nan
        return np.sqrt(np.sum(diff_vector * diff_vector))

    def _calc_dist_matrix(self, chain_one, chain_two):
        """Returns a matrix of C-alpha distances between two chains"""

        dist_atoms = 'CB'

        answer = np.zeros((len(chain_one), len(chain_two)), np.float)
        for row, residue_one in enumerate(chain_one):
            for col, residue_two in enumerate(chain_two):
                answer[row,
                       col] = self._calc_residue_dist(residue_one, residue_two, dist_atoms=dist_atoms)
        return answer

    def get_dist_mat(self, force=False):

        target_file = os.path.join(self._target_dir,
                                   'target_pdb_v%s.pkl' % self.VERSION)

        if not self._is_modeller and not force:
            dm = pkl_load(target_file)
            if dm is not None:
                return dm

        poly = self._get_residues()  # Polypeptide.Polypeptide(self._bio_chain) # #

        dm = self._calc_dist_matrix(poly, poly)  # [self._aa_mask, :][:, self._aa_mask]
        if not self._is_modeller:
            pkl_save(target_file, dm)
        return dm

    def _validate_data(self, dm, angels, sequence):
        if dm is None or angels is None or sequence is None:
            return
        l_dm = dm.shape[0]
        l_angels = len(angels)
        l_sequence = len(sequence)
        if l_dm != l_angels or l_dm != l_sequence or l_angels != l_sequence:
            LOGGER.warning('Protein data do not match for %s' %
                           (self.protein + self.chain))

    def _get_cm(self, dm):
        if dm is None:
            return
        nan_inds = np.logical_or(np.isnan(dm), dm == self.NAN_VALUE)
        dm[nan_inds] = self.NAN_VALUE

        cm = (dm < self._THRESHOLD).astype(np.float32)

        cm[nan_inds] = self.NAN_VALUE

        return cm

    @property
    def dm(self):
        return self.get_dist_mat()

    @property
    def cm(self):
        return self._get_cm(self.dm)

# class ProteinV0:
#     NAN_VALUE = -1
#     _THRESHOLD = 8
#
#     def __init__(self, protein, chain, pdb_path=PATHS.pdb, modeller_n_struc=1, version=5):
#         self.VERSION = version
#         self.protein = protein if type(protein) == str else protein.decode(
#             "utf-8")
#         self.chain = chain.upper() if type(chain) == str else chain.decode("utf-8").upper()
#
#         target = self.protein + self.chain
#         self.target = target
#         self._target_dir = os.path.join(PATHS.msa_structures, target)
#
#         if not os.path.exists(self._target_dir):
#             os.mkdir(self._target_dir)
#
#         self.pdb_path = pdb_path if type(pdb_path) == str else pdb_path.decode(
#             "utf-8")
#
#         self._is_modeller = self.pdb_path == PATHS.modeller
#
#         self.pdb_fname = path.join(
#             self.pdb_path, 'pdb%s.ent' %
#                            self.protein) if not self._is_modeller else get_modeller_pdb_file(
#             target, n_struc=modeller_n_struc)
#
#         if not path.exists(self.pdb_fname):
#             if self._is_modeller:
#                 LOGGER.info('no modeller file found %s' % self.pdb_fname)
#                 return
#             success_code = self.get_pdb_file()
#             if success_code != 0:
#                 return
#         self.pdb_parser = PDBParser(PERMISSIVE=1)
#         # self.angels = self.get_torsion_angle()
#         self.sequence = self.get_chain_seq()
#         self.str_seq = "".join(self.sequence)
#
#         # self.full_sequence = self.get_chain_seq_full()
#         # self.str_seq_full = "".join(self.full_sequence)
#         self.secondary_structure = self._get_secondary_structure()
#
#     def _check_dm(self):
#         if np.isnan(self.dm).sum() > 0:
#             self.dm = self.get_dist_mat(force=True)
#             LOGGER.info(f'Number of nas in {self.protein + self.chain} is {np.isnan(self.dm).sum()}')
#
#     def get_pdb_file(self):
#
#         msg = "cannot find a pdb file named '" + self.pdb_fname + "'."
#         LOGGER.info(msg)
#
#         pdb_file_zipped = 'pdb%s.ent.gz' % self.protein
#
#         ftp_file = 'ftp://ftp.wwpdb.org/pub/pdb/data/structures/divided/pdb/%s/%s' % (
#             self.protein[1:3], pdb_file_zipped)
#
#         ftp_cmd = "wget %s -P %s -q" % (ftp_file, PATHS.pdb)
#
#         subprocess.Popen(ftp_cmd, shell=True, stdout=open(os.devnull, 'wb')).wait()
#         cmd2 = 'gunzip -f ' + path.join(PATHS.pdb, pdb_file_zipped)
#         if os.path.isfile(path.join(PATHS.pdb, pdb_file_zipped)):
#             success = subprocess.Popen(cmd2, shell=True).wait()
#         else:
#             success = 1
#         return success
#
#     @property
#     def _aa_mask(self):
#
#         structure = self.pdb_parser.get_structure(id='strcuture',
#                                                   file=self.pdb_fname)
#         model = structure[0]
#         chain = model[self.chain]
#
#         poly = Polypeptide.Polypeptide(chain)
#         aa_mask = [Polypeptide.is_aa(r) for r in poly]
#         return aa_mask
#
#     @property
#     def ss3(self):
#
#         mapping = {'H': 'H', "G": "H", 'I': 'H', 'E': 'E', 'B': "E", 'S': "C", 'T': 'C', 'C': "C", '-': '-'}
#
#         dssp_dict = dssp_dict_from_pdb_file(self.pdb_fname)[0]
#         ss_seq = np.array([mapping[dssp_dict[k][1]] for k in dssp_dict if k[0] == self.chain])
#
#         return ''.join(ss_seq)
#
#     def _get_secondary_structure(self):
#
#         ss_file = os.path.join(
#             self._target_dir, 'secondary_structure_v%s.pkl' % self.VERSION)
#
#         if os.path.isfile(ss_file):
#             ss = pkl_load(ss_file)
#             if ss.shape[0] == len(self.sequence):
#                 return ss
#
#         one_hot_dict = {}
#         ss_keys = ['H', 'B', 'E', 'G', 'I', 'T', 'S', '-']
#         for i, k in enumerate(ss_keys):
#             one_hot_dict[k] = np.zeros(shape=(len(ss_keys),))
#             one_hot_dict[k][i] = 1
#         try:
#             pdb_seq = ''.join(self.sequence)
#         except TypeError:
#             return
#         ss_nas = np.stack([one_hot_dict['-'] for i in range(len(pdb_seq))])
#
#         try:
#             dssp_dict = dssp_dict_from_pdb_file(self.pdb_fname)[0]
#         except Exception:
#             return ss_nas
#
#         dssp_seq_arr = np.array([dssp_dict[k][0] for k in dssp_dict if k[0] == self.chain])
#
#         inds = self._aa_mask[0:len(dssp_seq_arr)]
#
#         dssp_seq = "".join(dssp_seq_arr[inds])
#
#         if len(pdb_seq) != len(dssp_seq):
#             return ss_nas
#         ss = np.stack([one_hot_dict[dssp_dict[k][1]] for k in dssp_dict if k[0] == self.chain])[inds, ...]
#
#         pkl_save(ss_file, ss)
#         return ss
#
#     def get_chain_seq(self):
#
#         target_seq_file = os.path.join(
#             self._target_dir, 'target_sequence_v%s.pkl' % self.VERSION)
#         if os.path.isfile(target_seq_file) and not self._is_modeller:
#             return pkl_load(target_seq_file)
#
#         structure = self.pdb_parser.get_structure(id='strcuture',
#                                                   file=self.pdb_fname)
#         model = structure[0]
#         try:
#             chain = model[self.chain]
#         except KeyError:
#             return
#
#         poly = Polypeptide.Polypeptide(chain)
#         sequence = np.array(poly.get_sequence())
#         try:
#             clean_sequence = sequence[self._aa_mask]
#         except IndexError:
#             return
#         if not self._is_modeller:
#             pkl_save(target_seq_file, clean_sequence)
#         return clean_sequence
#
#     def get_chain_seq_full(self):
#
#         # target_seq_file = os.path.join(
#         #     self._target_dir, 'target_sequence_v%s.pkl' % self.VERSION)
#         # if os.path.isfile(target_seq_file) and not self._is_modeller:
#         #     return pkl_load(target_seq_file)
#
#         protein, chain_id = self.protein, self.chain
#         pdb_file = self.pdb_fname
#         chain = {record.id: record.seq for record in SeqIO.parse(pdb_file, 'pdb-seqres')}
#         query_chain_id = chain_id if chain_id in chain else f'{protein.upper()}:{chain_id}'
#         clean_sequence = np.array([aa for aa in chain[query_chain_id]])
#         #
#         # if not self._is_modeller:
#         #     pkl_save(target_seq_file, clean_sequence)
#         return clean_sequence
#
#     def get_torsion_angle(self):
#
#         target_file = os.path.join(self._target_dir,
#                                    'target_angels_v%s.pkl' % self.VERSION)
#         if os.path.isfile(target_file) and not self._is_modeller:
#             return pkl_load(target_file)
#
#         structure = self.pdb_parser.get_structure(id='strcuture',
#                                                   file=self.pdb_fname)
#         model = structure[0]
#         try:
#             chain = model[self.chain]
#         except KeyError:
#             return
#
#         poly = Polypeptide.Polypeptide(chain)
#         all_angles = np.array(poly.get_phi_psi_list())
#         angels_arr = np.array(all_angles[self._aa_mask], dtype=np.float32)
#         if not self._is_modeller:
#             pkl_save(target_file, angels_arr)
#         return angels_arr
#
#     def _calc_residue_dist(self, residue_one, residue_two, dist_atoms='CA'):
#         """Returns the C-alpha distance between two residues"""
#
#         if not Polypeptide.is_aa(residue_one) or not Polypeptide.is_aa(residue_two):
#             return np.nan
#         dist_atom_1 = dist_atoms if dist_atoms in residue_one else 'CA'
#         dist_atom_2 = dist_atoms if dist_atoms in residue_two else 'CA'
#
#         try:
#             diff_vector = residue_one[dist_atom_1].coord - residue_two[dist_atom_2].coord
#         except KeyError:
#             return np.nan
#         return np.sqrt(np.sum(diff_vector * diff_vector))
#
#     def _calc_dist_matrix(self, chain_one, chain_two):
#         """Returns a matrix of C-alpha distances between two chains"""
#
#         dist_atoms = 'CB'
#
#         answer = np.zeros((len(chain_one), len(chain_two)), np.float)
#         for row, residue_one in enumerate(chain_one):
#             for col, residue_two in enumerate(chain_two):
#                 answer[row,
#                        col] = self._calc_residue_dist(residue_one, residue_two, dist_atoms=dist_atoms)
#         return answer
#
#     def get_dist_mat(self, force=False):
#
#         target_file = os.path.join(self._target_dir,
#                                    'target_pdb_v%s.pkl' % self.VERSION)
#
#         if not self._is_modeller and not force:
#             dm = pkl_load(target_file)
#             if dm is not None:
#                 return dm
#
#         structure = self.pdb_parser.get_structure(self.protein, self.pdb_fname)
#         model = structure[0]
#
#         try:
#
#             chain = model[self.chain]
#         except KeyError:
#             return
#
#         poly = Polypeptide.Polypeptide(chain)
#
#         dm = self._calc_dist_matrix(poly,
#                                     poly)[self._aa_mask, :][:, self._aa_mask]
#         if not self._is_modeller:
#             pkl_save(target_file, dm)
#         return dm
#
#     def _validate_data(self, dm, angels, sequence):
#         if dm is None or angels is None or sequence is None:
#             return
#         l_dm = dm.shape[0]
#         l_angels = len(angels)
#         l_sequence = len(sequence)
#         if l_dm != l_angels or l_dm != l_sequence or l_angels != l_sequence:
#             LOGGER.warning('Protein data do not match for %s' %
#                            (self.protein + self.chain))
#
#     def _get_cm(self, dm):
#         if dm is None:
#             return
#         nan_inds = np.logical_or(np.isnan(dm), dm == self.NAN_VALUE)
#         dm[nan_inds] = self.NAN_VALUE
#
#         cm = (dm < self._THRESHOLD).astype(np.float32)
#
#         cm[nan_inds] = self.NAN_VALUE
#
#         return cm
#
#     @property
#     def dm(self):
#         return self.get_dist_mat()
#
#     @property
#     def cm(self):
#         return self._get_cm(self.dm)
