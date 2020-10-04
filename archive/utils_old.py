import os

import numpy as np
import pandas as pd
import yaml
from numba import njit, jit
from os import path
from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import Polypeptide, is_aa
import pickle
import time
import logging

from .globals import MODELLER_PATH, LOCAL

LOGGER = logging.getLogger(__name__)

MODELLER_VERSION = 3
VERSION = 3


def get_modeller_pdb_file(target, n_struc=None, version=None):
    protein, chain = target[0:4], target[4]

    target_path = os.path.join(MODELLER_PATH, protein + chain)

    version = MODELLER_VERSION if version is None else version

    struc_num = '' if n_struc is None else n_struc

    target_pdb_fname = 'v%s_pdb' % version + protein + f'{struc_num}.ent'

    pdb_file_path = os.path.join(target_path, target_pdb_fname)

    return pdb_file_path


def pkl_save(filename, data):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)


def pkl_load(filename):
    if not os.path.isfile(filename):
        return
    if os.path.getsize(filename) == 0:
        os.remove(filename)
        return
    with open(filename, 'rb') as file:
        data = pickle.load(file, encoding='latin1')
    return data


def create_sifts_mapping():
    if LOCAL:
        return

    working_dir = '/cs/zbio/orzuk/projects/ContactMaps/data/MSA_Completion'
    if not path.isfile(path.join(working_dir, 'sifts_mapping.pkl')):
        sifts_mapping = {}
        sifts = pd.read_csv('pdb_chain_uniprot.csv')
        sifts_2 = pd.read_csv('pdb_chain_uniprot_plus.csv')
        sifts = pd.concat([sifts, sifts_2], ignore_index=True, axis=0)
        for index, row in sifts.iterrows():
            uniprot = row['SP_PRIMARY']
            uniprot_ind = (row['SP_BEG'], row['SP_END'])
            pdb = str(row['PDB']) + str(row['CHAIN'])
            pdb_ind = (row['RES_BEG'], row['RES_END'])
            mapping_data = (pdb, {'uniprot': uniprot_ind, 'pdb': pdb_ind})

            if uniprot not in sifts_mapping.keys():
                sifts_mapping[uniprot] = [mapping_data]
            else:
                prots = [f[0] for f in sifts_mapping[uniprot]]
                if pdb not in prots:
                    sifts_mapping[uniprot].append(mapping_data)
        pkl_save(filename=path.join(working_dir, 'sifts_mapping.pkl'),
                 data=sifts_mapping)
    else:
        sifts_mapping = pkl_load(path.join(working_dir, 'sifts_mapping.pkl'))
    return sifts_mapping


# sifts_mapping = create_sifts_mapping()


def calc_residue_dist(residue_one, residue_two):
    """Returns the C-alpha distance between two residues"""
    diff_vector = residue_one["CA"].coord - residue_two["CA"].coord
    return np.sqrt(np.sum(diff_vector * diff_vector))


def modeller_get_chain_seqs(target_protein, target_chain, version):
    target_path = path.join(MODELLER_PATH, target_protein + target_chain)
    target_pdb_fname = 'v%s_pdb' % version + target_protein + '.ent'

    pdb_file_path = path.join(target_path, target_pdb_fname)
    if not path.isfile(pdb_file_path):
        LOGGER.warning('File %s not found' % pdb_file_path)
        return None, None
    parser = PDBParser(PERMISSIVE=1, QUIET=True)
    structure_id = path.basename(target_pdb_fname).split('.')[0]
    try:
        structure = parser.get_structure(structure_id, pdb_file_path)
    except:
        print(
            "ERROR: failed parser.get_structure(structure_id, pdb_fname) for "
            + target_pdb_fname)
        return None
    model = structure[0]
    try:
        chain = model[target_chain]
    except KeyError:
        return None
    chain_lst = []
    for res in chain.get_residues():
        if is_aa(res) and res.get_id()[0] == ' ':
            if res.resname == 'UNK' or res.resname == 'ASX':
                chain_lst.append('-')
            elif res.resname == 'SEC':
                chain_lst.append('U')
            else:
                chain_lst.append(Polypeptide.three_to_one(res.resname))

    return chain_lst, chain


def modeller_calc_dist_matrix(target, chain_str, version):
    """Returns a matrix of C-alpha distances between two chains"""

    chain_seq, chain = modeller_get_chain_seqs(target, chain_str, version)
    if chain_seq is None:
        return
    L = len(chain_seq)
    dm = np.empty((L, L), np.float)
    dm[:] = np.nan
    e_chain = list(enumerate(chain))
    for i in range(L):
        row, residue_one = e_chain[i]
        if residue_one.resname == 'HOH':
            continue

        for j in range(i, L):
            col, residue_two = e_chain[j]
            if residue_two.resname == 'HOH':
                continue

            try:
                d = calc_residue_dist(residue_one, residue_two)

                dm[row, col] = d
                dm[col, row] = d

            except KeyError:
                dm[row, col] = np.nan
                dm[col, row] = np.nan
    return pd.DataFrame(dm, index=chain_seq, columns=chain_seq)


def uniprot_frequency():
    aa_freq_df = pd.DataFrame(
        {
            'A': 9.23,
            'Q': 3.77,
            'L': 9.90,
            'S': 6.63,
            'R': 5.77,
            'E': 6.16,
            'K': 4.90,
            'T': 5.55,
            'N': 3.81,
            'G': 7.35,
            'M': 2.37,
            'W': 1.30,
            'D': 5.48,
            'H': 2.19,
            'F': 3.91,
            'Y': 2.90,
            'C': 1.19,
            'I': 5.65,
            'P': 4.87,
            'V': 6.92,
            'X': 0.04
        },
        index=['frequency'])
    aa_freq_array = aa_freq_df.values / float(aa_freq_df.sum(axis=1))
    aa_pairwise_freq = np.matmul(aa_freq_array.transpose(), aa_freq_array)
    return pd.DataFrame(aa_pairwise_freq,
                        columns=aa_freq_df.keys(),
                        index=aa_freq_df.keys())


aa = list('-ACDEFGHIKLMNPQRSTVWYX')
aa_dict = {aa[i]: i for i in range(len(aa))}
aa_dict['Z'] = 0
aa_dict['B'] = 0
aa_dict['U'] = 0
aa_dict['O'] = 0


def get_local_sequence_identity_distmat(seq1, seq2, local_range):
    def _numeric_prot(prot):
        num_prot = np.vectorize(aa_dict.__getitem__)(prot).astype(np.int32)

        return num_prot

    return generate_local_sequence_distance_mat(_numeric_prot(seq1),
                                                _numeric_prot(seq2),
                                                local_range)


@njit
def generate_local_sequence_distance_mat(seq1, seq2, local_range):
    l = len(seq1)
    local_distance_mat = np.zeros(shape=(l, l))

    for i in range(l):
        for j in range(i, l):
            i_start = max(0, i - local_range)
            i_end = i + local_range

            j_start = max(0, j - local_range)
            j_end = j + local_range

            i_dist = np.mean(seq1[i_start:i_end] != seq2[i_start:i_end])
            j_dist = np.mean(seq1[j_start:j_end] != seq2[j_start:j_end])

            dist = i_dist + j_dist

            local_distance_mat[i, j] = dist
            local_distance_mat[j, i] = dist
    return local_distance_mat


@njit
def compute_pssm(numeric_msa):
    l = numeric_msa.shape[1]
    n = numeric_msa.shape[0]
    pssm = np.zeros(shape=(l, l, 44))
    for i in range(l):
        for j in range(i, l):
            pssm_i = [0] * 22
            pssm_j = [0] * 22
            col_j = numeric_msa[:, j]
            col_i = numeric_msa[:, i]

            for k in range(n):
                pssm_j[col_j[k]] += 1
                pssm_i[col_i[k]] += 1
            pssm_i_j = list(pssm_i) + list(pssm_j)
            pssm[i, j, :] = pssm_i_j
            pssm[j, i, :] = pssm_i_j

    return np.log(pssm)


def write_fasta(sequences, filename):
    """Saves list of sequence to fasta file

    Args:
        sequences (list[SeqRecord]): list of sequences
        filename (str): file name to save

    """
    with open(filename, "w") as output_handle:
        SeqIO.write(sequences, output_handle, "fasta")


def read_fasta(filename):
    with open(filename, "r") as output_handle:
        return SeqIO.parse(output_handle, "fasta")


def convert_to_aln(msa_filename, f_out):
    with open(f_out, 'w') as f:
        for record in SeqIO.parse(msa_filename, 'fasta'):
            f.write(str(record.seq) + '\n')


@njit
def compute_seq_indentity(target, reference):
    len_seq_1 = 0
    len_seq_2 = 0
    similarity = 0
    for i in range(len(target)):
        seq_1_i = target[i]
        seq_2_i = reference[i]
        seq_1_i_not_gap = seq_1_i != '-'
        seq_2_i_not_gap = seq_2_i != '-'

        if seq_1_i_not_gap or seq_2_i_not_gap:
            if seq_1_i == seq_2_i:
                similarity += 1
            if seq_1_i_not_gap:
                len_seq_1 += 1
            if seq_2_i_not_gap:
                len_seq_2 += 1
    sequence_identity = similarity / len_seq_1
    return sequence_identity


@njit
def compute_structures_identity_matrix(msa_seqs, structures, target):
    msa_size = len(msa_seqs)
    structures_size = len(structures)
    identity_matrix = np.zeros(shape=(structures_size, msa_size))
    for i in range(msa_size):
        for j in range(structures_size):
            seq = msa_seqs[i]
            struc = structures[j]
            if struc == target:
                seq_indentity = compute_seq_indentity(struc, seq)
            else:
                seq_indentity = compute_seq_indentity(seq, struc)
            identity_matrix[j, i] = seq_indentity
    return identity_matrix


def yaml_save(filename, data):
    with open(filename, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)


def yaml_load(filename):
    with open(filename, 'r') as stream:
        data_loaded = yaml.load(stream)

    return data_loaded


def parse_rr_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    def _is_dist_line(l):
        split = l.split(' ')
        if len(split) < 2:
            return False
        l1 = len(split[0])
        l2 = len(split[1])
        if l1 >= 1 and l2 <= 4 and l2 >= 1 and l2 <= 4:
            return True
        return False

    cols = ['X1', 'X2', 't0', 't1', 'prob']

    df = pd.DataFrame([x.strip('\n').split(' ') for x in lines[1:] if _is_dist_line(x)], columns=cols, dtype=np.float32)
    l = int(df.X2.max())

    cm = np.zeros((l, l))
    x_coords = np.array(df.X1.values, dtype=np.int32) - 1
    y_coords = np.array(df.X2.values, dtype=np.int32) - 1

    cm[x_coords, y_coords] = df.prob
    cm += cm.T

    return cm


def timefunc(f):
    def f_timer(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        duration = np.round(end - start, 2)
        if duration > 1:
            LOGGER.info("".join([f.__name__, ' took ',
                                 str(duration), ' time']))
        return result

    return f_timer

# def replace_with_dict_wrapper(pairs_alignment, aa_numeric_dict, log_ratio_dict):
#     k_preserved = np.array(
#         [np.float64(aa_numeric_dict[a[0]] + aa_numeric_dict[a[1]]) for a in log_ratio_dict.keys()],
#         dtype=np.float64)
#     v_preserved = np.array(list(log_ratio_dict.values()), dtype=np.float64)
#
#     pairs_log_ratio = replace_with_dict(pairs_alignment, k_preserved, v_preserved)
#     return pairs_log_ratio
