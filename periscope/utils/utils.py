import os
import subprocess
import tempfile
import warnings

import numpy as np
import pandas as pd
import yaml
from Bio.Seq import Seq
# from numba import njit
# from numba.typed import List
from os import path
from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import Polypeptide, is_aa, PDBIO, Select, PDBParser

import pickle
import time
import logging

from .constants import PATHS, LOCAL, DATASETS, DATASETS_FULL, N_REFS

LOGGER = logging.getLogger(__name__)
warnings.simplefilter('ignore', yaml.YAMLLoadWarning)

MODELLER_VERSION = 4
VERSION = 3


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        LOGGER.info('%r  %2.2f ms' % \
                    (method.__name__, (te - ts) * 1000))
        return result

    return timed


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
    score : str, optional (default: True)
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


def get_sotred_ec(raw_ec):
    # gets the sorted indices for ec

    raw_ec_local = raw_ec.copy()

    ec_no_na = raw_ec_local.dropna()

    ec_no_dup = ec_no_na.drop_duplicates('i')
    ec_sorted_i = ec_no_dup.sort_values('i')
    ec_sorted_j = ec_no_dup.sort_values('j')

    return ec_sorted_i, ec_sorted_j, ec_no_na


def save_chain_pdb(target, fname, pdb_fname, ind, skip_chain=False, old=True):
    pdb_parser = PDBParser(PERMISSIVE=1)
    s = pdb_parser.get_structure(target[0:4], pdb_fname)
    model = s[0]
    chains = model.child_list
    if len(chains) == 1 and chains[0].get_id() == ' ':
        chains[0].id = target[-1]
    io = PDBIO()
    io.set_structure(s)
    tmp_file = os.path.join(os.getcwd(), f'{target}_tmp.pdb')

    class ChainSelect(Select):
        def __init__(self, chain, skip_chain, old=True):
            self._chain = chain
            self._skip_chain = skip_chain
            self._old = old

        def accept_chain(self, chain):
            if chain.id == self._chain or self._skip_chain:
                return 1
            else:
                return 0

        def accept_residue(self, residue):
            if residue.full_id[3][0] == ' ' or residue.full_id[3][0] == 'H_MSE':
                return 1
            elif self._old and residue.full_id[3][0] != 'W':
                return 1
            else:
                return 0

    io.save(tmp_file, ChainSelect(target[4], skip_chain=skip_chain, old=old))
    reres_cmd = f'pdb_reres -{ind} {tmp_file} > {fname}'
    subprocess.call(reres_cmd, shell=True)
    os.remove(tmp_file)

    # if not skip_chain:
    #     io.save(tmp_file, ChainSelect(target[4]))
    #     reres_cmd = f'pdb_reres -{ind} {tmp_file} > {fname}'
    #     subprocess.call(reres_cmd, shell=True)
    #     os.remove(tmp_file)
    # else:
    #     shutil.copy(pdb_fname, fname)


def get_modeller_pdb_file(target, n_struc=None, templates=False, sp=False):
    protein, chain = target[0:4], target[4]

    target_path = os.path.join(PATHS.modeller, protein + chain)
    if templates:
        target_path = os.path.join(PATHS.modeller, 'templates', protein + chain)
    if sp:
        target_path = os.path.join(PATHS.modeller, 'templates', 'sp', protein + chain)

    version = MODELLER_VERSION

    struc_num = '' if n_struc is None else n_struc

    target_pdb_fname = 'v%s_pdb' % version + protein + f'{struc_num}.ent'

    pdb_file_path = os.path.join(target_path, target_pdb_fname)

    return pdb_file_path


def np_to_csv(filename, data):
    pd.DataFrame(data).to_csv(filename)


def np_read_csv(filename):
    return pd.read_csv(filename, index_col=0).values


def read_fasta(filename, full=False):
    fasta = list(SeqIO.parse(open(filename, "r"), "fasta"))
    if len(fasta) == 0:
        return
    if full:
        return fasta
    return fasta[0]


def _get_uniprot_seq(uniprot):
    baseUrl = "http://www.uniprot.org/uniprot/"
    currentUrl = baseUrl + uniprot + ".fasta"
    file = 'tmp2.fasta'
    cmd = f'wget {currentUrl} -q -O {file}'

    msg = subprocess.call(cmd, shell=True)
    if msg != 0:
        return
    Seq = read_fasta(file)
    os.remove(file)
    return Seq


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


def get_target_dataset(target):
    datasets = 'train eval pfam cameo membrane cameo41'.split(' ')
    for d in datasets:
        if target in getattr(DATASETS, d):
            return d


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def run_clustalo(sequences, fname, target=None, structures=None):
    SeqIO.write(sequences, fname, 'fasta')

    if structures is None:
        cmd = f'clustalo -i {fname} -o {fname} --force'
        subprocess.run(cmd, shell=True)
        return

    msa_file = get_aln_fasta(target)
    msa = read_fasta(msa_file, full=True)

    def _get_id(h):
        if len(h.id) == 5:
            return h.id
        return h.id.split('|')[1]

    msa_short = [msa[0]] + [h for h in msa if _get_id(h) in structures]
    valid_inds = np.array(list(msa[0].seq)) != '-'
    for h in msa_short:
        h.seq = Seq("".join(np.array(list(h.seq))[valid_inds]))

    with tempfile.NamedTemporaryFile(suffix='.fasta') as f:

        SeqIO.write(msa_short, f.name, 'fasta')
        cmd = f'clustalo -i {fname} --p1 {f.name} -o {fname} --force'
        p = subprocess.run(cmd, shell=True)
        success = p.returncode
        if success != 0:
            cmd = f'clustalo -i {fname} -o {fname} --force'
            subprocess.run(cmd, shell=True)

        aln_seqs = read_fasta(f'{fname}', full=True)[0:len(sequences)]
        SeqIO.write(aln_seqs, fname, 'fasta')


def create_sifts_mapping():
    if LOCAL:
        return

    working_dir = PATHS.data
    if not path.isfile(path.join(PATHS.data, 'sifts_mapping.pkl')):
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
    target_path = path.join(PATHS.modeller, target_protein + target_chain)
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


def get_dist_cat_mat(l):
    seq_dist_mat = compute_sequence_distance_mat_np(l)
    invalid_inds = seq_dist_mat < 6
    s_inds = np.logical_and(seq_dist_mat >= 6, seq_dist_mat < 12)
    m_inds = np.logical_and(seq_dist_mat >= 12, seq_dist_mat < 24)
    l_inds = seq_dist_mat >= 24

    seq_dist_mat[invalid_inds] = 0
    seq_dist_mat[s_inds] = 1
    seq_dist_mat[m_inds] = 2
    seq_dist_mat[l_inds] = 3

    return seq_dist_mat


def compute_sequence_distance_mat_np(l):
    l_int = int(l)

    def _i_minus_j(i, j):
        return np.abs(i - j)

    sequence_distance = np.fromfunction(_i_minus_j,
                                        shape=(l_int, l_int),
                                        dtype=np.float32)
    lower_triu = np.tril_indices(l)
    sequence_distance[lower_triu] = 0

    return sequence_distance


def get_local_sequence_identity_distmat(seq1, seq2, local_range):
    def _numeric_prot(prot):
        num_prot = np.vectorize(aa_dict.__getitem__)(prot).astype(np.int32)

        return num_prot

    return generate_local_sequence_distance_mat(_numeric_prot(seq1),
                                                _numeric_prot(seq2),
                                                local_range)


# @njit
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


# @njit
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


def convert_to_aln(msa_filename, f_out):
    with open(f_out, 'w') as f:
        for record in SeqIO.parse(msa_filename, 'fasta'):
            f.write(str(record.seq) + '\n')


# @njit
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
    sequence_identity = similarity / min(len_seq_1, len_seq_2)
    return sequence_identity


def compute_structures_identity_matrix(msa_seqs, structures, target):
    typed_msa_seqs = list()
    [typed_msa_seqs.append(x) for x in msa_seqs]
    typed_structures = list()
    [typed_structures.append(x) for x in structures]
    return _compute_structures_identity_matrix(typed_msa_seqs, typed_structures, target)


# @njit
def _compute_structures_identity_matrix(msa_seqs, structures, target):
    msa_size = len(msa_seqs)
    structures_size = len(structures)
    identity_matrix = np.zeros(shape=(structures_size, msa_size))
    for i in range(msa_size):
        for j in range(structures_size):
            seq = msa_seqs[i]
            struc = structures[j]
            if struc == target:
                seq_identity = compute_seq_indentity(struc, seq)
            else:
                seq_identity = compute_seq_indentity(seq, struc)
            identity_matrix[j, i] = seq_identity
    return identity_matrix


def yaml_save(filename, data):
    with open(filename, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)


def yaml_load(filename):
    with open(filename, 'r') as stream:
        data_loaded = yaml.load(stream)

    return data_loaded


def get_fasta_fname(target, full):
    if full:
        return os.path.join(PATHS.msa, 'query', target + '_full.fasta')
    return os.path.join(get_target_path(target), f'{target}.fasta')


def get_target_path(target, new=False):
    f_name = target + '_new' if new else target
    t_path = os.path.join(PATHS.proteins, target[1:3], f_name)
    check_path(t_path)
    return t_path


def get_pdb_fname(protein):
    return os.path.join(PATHS.data, 'pdb', protein[1:3], f'pdb{protein}.ent')


def get_predicted_pdb(model, target, sswt=5, selectrr='2.0L'):
    dataset = get_target_dataset(target)
    outdir = os.path.join(model.path, f'cns_{sswt}_{selectrr.replace(".", "_")}', dataset, target)
    predicted_pdb = os.path.join(outdir, 'stage1', f'{target}_model1.pdb')
    return predicted_pdb


def get_target_scores_file(target):
    return os.path.join(get_target_path(target), "scores.pkl")


def get_target_ccmpred_file(target):
    ccmpred_path = os.path.join(get_target_path(target), 'ccmpred')
    check_path(ccmpred_path)

    ccmpred_file = os.path.join(ccmpred_path, f'{target}.mat')

    return ccmpred_file


def get_target_evfold_file(target):
    return os.path.join(get_target_path(target), 'evfold', f'{target}_v2.pkl')


def get_aln_fasta(target):
    target_hhblits_path = os.path.join(get_target_path(target), 'hhblits')
    fasta_file = os.path.join(target_hhblits_path, f'{target}_v2.fasta')
    return fasta_file


def get_clustalo_aln(target, n_refs=N_REFS):
    features_path = os.path.join(get_target_path(target), 'features')
    clustalo_path = os.path.join(features_path, 'clustalo')
    return os.path.join(clustalo_path, f'aln_r_{n_refs}.fasta')


def get_a3m_fname(target):
    target_hhblits_path = os.path.join(get_target_path(target), 'hhblits')
    a3m_file = os.path.join(target_hhblits_path, f'{target}.a3m')
    return a3m_file


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


def csv_to_list(filename):
    data = pd.read_csv(filename, index_col=0)
    return list(data.iloc[:, 0])


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
