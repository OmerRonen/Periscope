import logging
import os
import subprocess
from os import path

from Bio.PDB.DSSP import dssp_dict_from_pdb_file

import numpy as np
from Bio.PDB import Polypeptide, DSSP
from Bio.PDB.PDBParser import PDBParser
from .globals import PDB_PATH, MSA_STRUCTURES_DATA_PATH, MODELLER_PATH
from .utils_old import pkl_save, pkl_load, get_modeller_pdb_file

LOGGER = logging.getLogger(__name__)


class Protein:

    def __init__(self, protein, chain, pdb_path=PDB_PATH, modeller_n_struc=None, version=3):
        self.VERSION = version
        self.protein = protein if type(protein) == str else protein.decode(
            "utf-8")
        self.chain = chain if type(chain) == str else chain.decode("utf-8")

        target = self.protein + self.chain
        self.target = target
        self._target_dir = os.path.join(MSA_STRUCTURES_DATA_PATH, target)

        if not os.path.exists(self._target_dir):
            os.mkdir(self._target_dir)

        self.pdb_path = pdb_path if type(pdb_path) == str else pdb_path.decode(
            "utf-8")

        self._is_modeller = self.pdb_path == MODELLER_PATH

        self.pdb_fname = path.join(
            self.pdb_path, 'pdb%s.ent' %
                           self.protein) if not self._is_modeller else get_modeller_pdb_file(
            target, n_struc=modeller_n_struc)

        if not path.exists(self.pdb_fname):
            if self._is_modeller:
                LOGGER.info('no modeller file found %s' % self.pdb_fname)
                return
            self.get_pdb_file()
        self.pdb_parser = PDBParser(PERMISSIVE=0)
        self.dm = self.get_dist_mat()
        self.angels = self.get_torsion_angle()
        self.sequence = self.get_chain_seq()
        self.secondary_structure = self._get_secondary_structure()
        self._validate_data(self.dm, self.angels, self.sequence)

    def _check_dm(self):
        if np.isnan(self.dm).sum() > 0:
            self.dm = self.get_dist_mat(force=True)
            LOGGER.info(f'Number of nas in {self.protein + self.chain} is {np.isnan(self.dm).sum()}')

    def get_pdb_file(self):

        msg = "cannot find a pdb file named '" + self.pdb_fname + "'."
        LOGGER.info(msg)

        pdb_file_zipped = 'pdb%s.ent.gz' % self.protein

        ftp_file = 'ftp://ftp.wwpdb.org/pub/pdb/data/structures/divided/pdb/%s/%s' % (
            self.protein[1:3], pdb_file_zipped)

        ftp_cmd = "wget %s -P %s" % (ftp_file, PDB_PATH)

        subprocess.Popen(ftp_cmd, shell=True).wait()
        cmd2 = 'gunzip -f ' + path.join(PDB_PATH, pdb_file_zipped)
        subprocess.Popen(cmd2, shell=True).wait()

    @property
    def _aa_mask(self):

        structure = self.pdb_parser.get_structure(id='strcuture',
                                                  file=self.pdb_fname)
        model = structure[0]
        chain = model[self.chain]

        poly = Polypeptide.Polypeptide(chain)
        aa_mask = [Polypeptide.is_aa(r) for r in poly]
        return aa_mask

    def _get_secondary_structure(self):

        ss_file = os.path.join(
            self._target_dir, 'secondary_structure_v%s.pkl' % self.VERSION)

        if os.path.isfile(ss_file):
            ss = pkl_load(ss_file)
            if ss.shape[0] == len(self.sequence):
                return ss

        one_hot_dict = {}
        ss_keys = ['H', 'B', 'E', 'G', 'I', 'T', 'S', '-']
        for i, k in enumerate(ss_keys):
            one_hot_dict[k] = np.zeros(shape=(len(ss_keys),))
            one_hot_dict[k][i] = 1

        pdb_seq = ''.join(self.sequence)
        ss_nas = np.stack([one_hot_dict['-'] for i in range(len(pdb_seq))])

        try:
            dssp_dict = dssp_dict_from_pdb_file(self.pdb_fname)[0]
        except Exception:
            return ss_nas

        dssp_seq_arr = np.array([dssp_dict[k][0] for k in dssp_dict if k[0] == self.chain])

        inds = self._aa_mask[0:len(dssp_seq_arr)]

        dssp_seq = "".join(dssp_seq_arr[inds])

        if len(pdb_seq) != len(dssp_seq):
            return ss_nas
        ss = np.stack([one_hot_dict[dssp_dict[k][1]] for k in dssp_dict if k[0] == self.chain])[inds, ...]

        pkl_save(ss_file, ss)
        return ss

    def get_chain_seq(self):

        target_seq_file = os.path.join(
            self._target_dir, 'target_sequence_v%s.pkl' % self.VERSION)
        if os.path.isfile(target_seq_file) and not self._is_modeller:
            return pkl_load(target_seq_file)

        structure = self.pdb_parser.get_structure(id='strcuture',
                                                  file=self.pdb_fname)
        model = structure[0]
        chain = model[self.chain]

        poly = Polypeptide.Polypeptide(chain)
        sequence = np.array(poly.get_sequence())
        clean_sequence = sequence[self._aa_mask]
        if not self._is_modeller:
            pkl_save(target_seq_file, clean_sequence)
        return clean_sequence

    def get_torsion_angle(self):

        target_file = os.path.join(self._target_dir,
                                   'target_angels_v%s.pkl' % self.VERSION)
        if os.path.isfile(target_file) and not self._is_modeller:
            return pkl_load(target_file)

        structure = self.pdb_parser.get_structure(id='strcuture',
                                                  file=self.pdb_fname)
        model = structure[0]
        chain = model[self.chain]

        poly = Polypeptide.Polypeptide(chain)
        all_angles = np.array(poly.get_phi_psi_list())
        angels_arr = np.array(all_angles[self._aa_mask], dtype=np.float32)
        if not self._is_modeller:
            pkl_save(target_file, angels_arr)
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
        if self.VERSION == 2:
            dist_atoms = 'CA'
        elif self.VERSION == 3:
            dist_atoms = 'CB'
        elif self.VERSION == 4:
            dist_atoms = 'C'

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

        structure = self.pdb_parser.get_structure(self.protein, self.pdb_fname)
        model = structure[0]

        chain = model[self.chain]

        poly = Polypeptide.Polypeptide(chain)

        dm = self._calc_dist_matrix(poly,
                                    poly)[self._aa_mask, :][:, self._aa_mask]
        if not self._is_modeller:
            pkl_save(target_file, dm)
        return dm

    def _validate_data(self, dm, angels, sequence):
        l_dm = dm.shape[0]
        l_angels = len(angels)
        l_sequence = len(sequence)
        if l_dm != l_angels or l_dm != l_sequence or l_angels != l_sequence:
            LOGGER.warning('Protein data do not match for %s' %
                           (self.protein + self.chain))
