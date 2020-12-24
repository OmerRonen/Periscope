import itertools
import os
import re
import tempfile

import numpy as np
import pandas as pd
from Bio import pairwise2, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from ..utils.constants import PROTEIN_BOW_DIM
from ..utils.protein import Protein
from ..utils.utils import (get_family_path, compute_structures_identity_matrix,
                           get_aln_fasta, get_target_path, pkl_load,
                           pkl_save, check_path, run_clustalo, read_fasta, create_sifts_mapping)

aa = list('-ACDEFGHIKLMNPQRSTVWYX')
aa_dict = {aa[i]: i for i in range(len(aa))}
aa_dict['Z'] = 0
aa_dict['B'] = 0
aa_dict['U'] = 0
aa_dict['O'] = 0

SIFTS_MAPPING = create_sifts_mapping()


def _slice_rows_cols(inds, arr):
    row_idx = np.array(inds)
    col_idx = np.array(inds)
    slice_arr = arr[row_idx[:, None], col_idx, ...]
    return slice_arr


def _slice_rows(rows, arr):
    row_idx = np.array(rows)
    slice_arr = arr[row_idx, ...]
    return slice_arr


def _is_all_gaps(seq_msa):
    arr_seq = np.array(list(seq_msa))
    all_gaps = np.mean(arr_seq == '-') == 1
    return all_gaps


def get_pdb_entries(uniprot):
    """Finds all pdb entries associates with a uniprot id

    Args:
        uniprot (str): uniprot id

    Returns:
        list[str]: list of pdb ids

    """
    if len(uniprot) == 5:
        return [uniprot]
    has_mapping = uniprot in SIFTS_MAPPING
    if not has_mapping:
        return []
    pdbs = [mapping[0] for mapping in SIFTS_MAPPING[uniprot]]
    return pdbs


def find_aligned_dm_uniprot(uniprot, seq_msa):
    """Calculates Distance matrix aligned with the msa sequence for a uniprot entry

    Args:
        uniprot (str): uniprot name of the homologous protein
        seq_msa (str): msa sequence of length L_msa

    Returns:
        np.array: of shape (L_msa, L_msa)

    """
    pdbs = get_pdb_entries(uniprot)
    for pdb in pdbs:
        hom_prot = Protein(pdb[0:4], pdb[4])

        seq_pdb = hom_prot.str_seq
        if seq_pdb is None or len(seq_pdb) == 0:
            continue
        is_aligned = find_aligned_mapping(seq_msa, seq_pdb)
        if is_aligned:
            return pdb

    return


def find_aligned_mapping(seq_msa, seq_pdb):
    """Finds matching indices between pdb and msa sequences

    Args:
        seq_msa (str): msa sequence, i.e containing gaps
        seq_pdb (str): pdb sequence

    Returns:
        bool: True if the sequences match

    """

    mismatch_penalty = -100 * len(seq_msa)
    seq_msa_arr = np.array(list(seq_msa))
    seq_msa_no_gaps = "".join(seq_msa_arr[seq_msa_arr != '-'])
    alignment = pairwise2.align.globalms(seq_msa_no_gaps, seq_pdb, 1, mismatch_penalty, -.5, -.1)[0]
    # print(format_alignment(*alignment))

    seq_msa_aligned = np.array(list(alignment.seqA))
    seq_pdb_aligned = np.array(list(alignment.seqB))
    # we force high mismatch penalty so all mush be matches
    msa_inds = np.where(seq_msa_aligned != '-')[0]
    aligned_pdb_seq = "".join(seq_pdb_aligned[msa_inds])
    # we want to make sure that we maintain the full sequence
    s = seq_pdb.find(aligned_pdb_seq)
    if s == -1:
        return False
    pdb_inds = list(range(s, s + len(aligned_pdb_seq)))

    assert "".join(np.array(list(seq_msa_aligned))[msa_inds]) == "".join(np.array(list(seq_pdb))[pdb_inds])

    return True


def _get_id_family(seq, target):
    if "UniRef100" in seq.name:
        return seq.name.split('_')[1], ""
    if len(seq.name) == 5:
        return seq.name, seq.name
    des = seq.description.split('|')
    uniprot_id = des[1]

    if len(des) == 3:
        pdb = ""
        return uniprot_id, pdb
    pdbs = des[3].split("+")
    if target in pdbs:
        return None, None
    pdb = pdbs[0].split('_')[0].lower() + pdbs[0].split('_')[1]

    return uniprot_id, pdb


def _get_id(seq, target):
    if seq.id == target:
        return seq.id, ""
    if "|" in seq.id:
        return seq.id.split('|')[1]
    return seq.id.split('_')[1], ""


def get_seq_dist_mat(target, clustalo_msa):
    structures_list = [f'{s.description}' for s in clustalo_msa]
    target_seq = [s for s in clustalo_msa if s.id == target][0].seq

    msa_structures_str = [str(s.seq) for s in clustalo_msa]

    id_mat = compute_structures_identity_matrix(msa_structures_str,
                                                msa_structures_str,
                                                target=str(target_seq.upper()))
    identity_mat = pd.DataFrame(id_mat,
                                columns=structures_list,
                                index=structures_list)

    return identity_mat


def _get_record(uniprot, pdb):
    prot = Protein(pdb[0:4], pdb[4])
    seq = prot.str_seq

    return SeqRecord(Seq(seq), id=pdb, name=uniprot, description='')


def _get_aln_ss_acc(msa_seq, pdb_id):
    pdb_ss_acc = Protein(pdb_id[0:4], pdb_id[4]).ss_acc
    ss_acc_arr = np.empty((len(msa_seq), pdb_ss_acc.shape[-1]))
    ss_acc_arr[:] = np.nan
    inds_aln = np.where(np.array(list(msa_seq)) != '-')[0]
    inds_pdb = np.array(list(range(len(inds_aln))))

    ss_acc_arr[inds_aln, ...] = pdb_ss_acc[inds_pdb, ...]

    return ss_acc_arr


def _one_hot_msa(numeric_msa):
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


def _get_aln_dm(msa_seq, pdb_id):
    dm = np.empty((len(msa_seq), len(msa_seq)))
    dm[:] = np.nan
    inds_aln = np.where(np.array(list(msa_seq)) != '-')[0]
    inds_pdb = np.array(list(range(len(inds_aln))))

    pairs_aln = list(itertools.combinations(inds_aln, 2))
    pairs_pdb = list(itertools.combinations(inds_pdb, 2))

    inds_aln = list(zip(*pairs_aln))
    inds_pdb = list(zip(*pairs_pdb))

    prot_dm = Protein(pdb_id[0:4], pdb_id[4]).dm
    dm[inds_aln[0], inds_aln[1]] = prot_dm[inds_pdb[0], inds_pdb[1]]
    dm[inds_aln[1], inds_aln[0]] = prot_dm[inds_pdb[1], inds_pdb[0]]

    return dm


class Aligner:
    """This class handles the aligning of msa sequence to a given target"""

    def __init__(self, target, family=None):
        """

        Args:
            target (str): pdb id of the target, including chain (i.e 5ms3A)
            family (str): name of the protein family
        """

        self.target = target
        self._family = family
        self._get_id = _get_id if self._family is None else _get_id_family

        self._msa_data_path = os.path.join(get_target_path(target, self._family), 'features')
        self._family_path = self._msa_data_path if self._family is None else get_family_path(self._family)
        self._msa_file = get_aln_fasta(self.target, self._family)

    def _parse_msa(self):

        fasta_seqs = list(SeqIO.parse(self._msa_file, "fasta"))
        sequences = {}
        for seq in fasta_seqs:
            uniprot_id, pdb_id = self._get_id(seq, self.target)
            if uniprot_id in sequences or uniprot_id is None:
                continue
            is_target = pdb_id == self.target
            id = pdb_id if is_target else uniprot_id
            is_pdb = len(id) == 5
            seq = Seq(str(seq.seq).upper()) if is_pdb else Seq(re.sub('[a-z]', '-', str(seq.seq)))
            s = SeqRecord(seq, id=id)  # SeqRecord(seq.seq, id=id)
            sequences[id] = s

        return sequences

    def get_ref_map(self):
        """Maps uniprot sequences to pdb's

        Args:
            target (str): pdb id of the target, including chain (i.e 5ms3A)
            msa (dict[str, Seq]): uniprot id to Seq dict

        Returns:
            dict[str, str]: mapping from uniprot to pdb id

        """
        msa = self._parse_msa()
        target = self.target

        ref_map = {}

        seq_msa_target = str(msa[target].seq)

        ref_map_file = os.path.join(self._family_path, self.target, f'ref_map.pkl')
        if self._family is None:
            ref_map_file = os.path.join(self._family_path,  f'ref_map.pkl')

        if os.path.isfile(ref_map_file):
            return pkl_load(ref_map_file)

        for uniprot in msa:
            seq_msa = str(msa[uniprot].seq)

            is_target = uniprot == target
            length_mismatch = len(seq_msa) != len(seq_msa_target)
            all_gaps = _is_all_gaps(seq_msa)

            if is_target or length_mismatch or all_gaps:
                continue
            pdb = find_aligned_dm_uniprot(uniprot, seq_msa)

            if pdb is None or pdb == self.target:
                continue

            ref_map[uniprot] = pdb

        pkl_save(ref_map_file, ref_map)

        return ref_map

    @property
    def n_homs(self):
        return len(self.get_structures_msa()) - 1

    @property
    def has_templates(self):
        has_tmplts = self.n_homs > 1
        return has_tmplts

    def _get_clustalo_msa_raw(self):
        clustalo_path = os.path.join(self._msa_data_path, 'clustalo')
        check_path(clustalo_path)
        fname = os.path.join(clustalo_path, f'aln.fasta')
        ref_map = self.get_ref_map()
        structures = list(ref_map.keys())

        ref_map[self.target] = self.target
        if not os.path.isfile(fname):
            sequences = [_get_record(u, p) for u, p in ref_map.items() if Protein(p[0:4], p[4]).str_seq is not None]
            if Protein(self.target[0:4], self.target[4]).str_seq is None:
                seq_full = np.array(list(self._parse_msa()[self.target]))
                seq = ''.join(seq_full[seq_full != '-'])
                sequences += [SeqRecord(Seq(seq), id=self.target, name=self.target, description='')]
            run_clustalo(sequences, fname, self.target, structures, self._family)
        aln = read_fasta(fname, True)

        ref_map_inv = {y: x for x, y in ref_map.items()}

        for h in aln:
            if h.id == self.target:
                continue
            h.description = ref_map_inv.get(h.id, h.id)
        return aln

    def get_structures_msa(self):
        """Returns the msa, aligned with all the known templates

        Returns:
            list: list of ordered valid sequneces

        """
        unfiltered_msa = self._get_clustalo_msa_raw()

        seq_id_mat = get_seq_dist_mat(self.target, unfiltered_msa)
        target_col_ind = np.where(seq_id_mat.index == self.target)[0][0]
        target_seq_id = seq_id_mat.iloc[target_col_ind,]
        thres = 1.1 if self._family == 'xcl1_family' else 0.95

        to_keep = np.logical_or(target_seq_id < thres, target_seq_id.index == self.target)
        filtered_msa = [s for i, s in enumerate(unfiltered_msa) if to_keep[i]]
        target_seq_id_filtered = target_seq_id[to_keep]

        arg_sort = np.array(target_seq_id_filtered.argsort()[::-1])
        filtered_msa_sorted = [filtered_msa[ind] for ind in arg_sort]

        return filtered_msa_sorted

    @property
    def closest_template(self):
        if not self.has_templates:
            return
        return self.get_structures_msa()[1].id

    @property
    def templates_distance_tensor(self):

        if not self.has_templates:
            return

        fname = os.path.join(self._msa_data_path, 'templates_distance_tensor.pkl')
        if os.path.isfile(fname):
            return pkl_load(fname)

        aln = self.get_structures_msa()
        dms = np.stack([_get_aln_dm(s.seq, s.id) for s in aln[1:]], axis=2)
        inds_target = np.where(np.array(list(aln[0].seq)) != '-')[0]
        dms_target = _slice_rows_cols(inds_target, dms)
        pkl_save(fname, dms_target)
        return dms_target

    @property
    def templates_sequence_tensor(self):

        if not self.has_templates:
            return

        fname = os.path.join(self._msa_data_path, 'templates_sequence_tensor.pkl')
        if os.path.isfile(fname):
            return pkl_load(fname)

        aln = self.get_structures_msa()

        aln_df = pd.DataFrame(aln[1:])

        numeric_msa = np.vectorize(aa_dict.__getitem__)(aln_df).astype(
            np.int32)
        inds_target = np.where(np.array(list(aln[0].seq)) != '-')[0]

        numeric_msa_target = numeric_msa[..., inds_target]

        seqs = _one_hot_msa(numeric_msa_target).transpose([1, 2, 0])
        pkl_save(fname, seqs)
        return seqs

    @property
    def templates_ss_acc_tensor(self):

        if not self.has_templates:
            return

        fname = os.path.join(self._msa_data_path, 'templates_ss_acc_tensor.pkl')
        if os.path.isfile(fname):
            return pkl_load(fname)

        aln = self.get_structures_msa()
        ss_acc = np.stack([_get_aln_ss_acc(s.seq, s.id) for s in aln[1:]], axis=2)
        inds_target = np.where(np.array(list(aln[0].seq)) != '-')[0]
        ss_acc_target = _slice_rows(inds_target, ss_acc)
        pkl_save(fname, ss_acc_target)

        return ss_acc_target

    @property
    def known_structures(self):
        return list(self.get_ref_map().keys())

    @property
    def templates_ss_acc_seq_tensor(self):
        if not self.has_templates:
            return
        return np.concatenate([self.templates_sequence_tensor,
                               self.templates_ss_acc_tensor], axis=1)
