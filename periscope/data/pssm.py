import os
import numpy as np
import pandas as pd
from numba import prange, njit
from scipy.interpolate import interp1d

# cores = 1  # Set number of CPUs with numba.
# os.environ["MKL_NUM_THREADS"] = "%s" % cores
# os.environ["NUMEXPR_NUM_THREADS"] = "%s" % cores
# os.environ["OMP_NUM_THREADS"] = "%s" % cores
# os.environ["OPENBLAS_NUM_THREADS"] = "%s" % cores
# os.environ["VECLIB_MAXIMUM_THREADS"] = "%s" % cores
# os.environ['NUMBA_DEFAULT_NUM_THREADS'] = "%s" % cores
# os.environ["NUMBA_NUM_THREADS"] = "%s" % cores
#
curr_float = np.float32
curr_int = np.int16

aa = list('-ACDEFGHIKLMNPQRSTVWYX')
aa_dict = {aa[i]: i for i in range(len(aa))}
aa_dict['Z'] = 0
aa_dict['B'] = 0
aa_dict['U'] = 0
aa_dict['O'] = 0
# aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
#       'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V',  'W', 'Y', '-']
# aadict = {aa[k]: k for k in range(len(aa))}
#
# aadict['X'] = len(aa)
# aadict['B'] = len(aa)
# aadict['Z'] = len(aa)
# aadict['O'] = len(aa)
# aadict['U'] = len(aa)


for key in ['a', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'y', 'x']:
    aa_dict[key] = aa_dict[key.upper()]
aa_dict['b'] = 0
aa_dict['z'] = 0
aa_dict['.'] = -1


def seq2num(string):
    if type(string) == str:
        return np.array([aa_dict[x] for x in string])[np.newaxis, :]
    elif type(string) == list:
        return np.array([[aa_dict[x] for x in string_] for string_ in string])


def num2seq(num):
    if num.ndim == 1:
        return ''.join([aa[min(x, len(aa) - 1)] for x in num])
    else:
        return [''.join([aa[min(x, len(aa) - 1)] for x in num_seq]) for num_seq in num]


def load_FASTA(filename, with_labels=False, remove_insertions=True, drop_duplicates=True):
    count = 0
    current_seq = ''
    all_seqs = []
    if with_labels:
        all_labels = []
    with open(filename, 'r') as f:
        for line in f:
            if line[0] == '>':
                all_seqs.append(current_seq)
                current_seq = ''
                if with_labels:
                    all_labels.append(line[1:].replace(
                        '\n', '').replace('\r', ''))
            else:
                current_seq += line.replace('\n', '').replace('\r', '')
                count += 1
            if remove_insertions:
                current_seq = ''.join(
                    [x for x in current_seq if not (x.islower() | (x == '.'))])

        all_seqs.append(current_seq)
        all_seqs = np.array(list(
            map(lambda x: [aa_dict[y] for y in x], all_seqs[1:])), dtype=curr_int, order="c")

    if drop_duplicates:
        all_seqs = pd.DataFrame(all_seqs).drop_duplicates()
        if with_labels:
            all_labels = np.array(all_labels)[all_seqs.index]
        all_seqs = np.array(all_seqs)

    if with_labels:
        return all_seqs, np.array(all_labels)
    else:
        return all_seqs


@njit#(parallel=False, cache=True, nogil=False)
def weighted_average(config, weights, q):
    B = config.shape[0]
    N = config.shape[1]
    out = np.zeros((N, q), dtype=curr_float)
    for b in prange(B):
        for n in prange(N):
            out[n, config[b, n]] += weights[b]
    out /= weights.sum()
    return out


@njit#(parallel=True, cache=True)
def count_neighbours(MSA, threshold=0.1, remove_gaps=False):  # Compute reweighting
    B = MSA.shape[0]
    num_neighbours = np.ones(B, dtype=curr_int)
    for b1 in prange(B):
        for b2 in prange(B):
            if b2 > b1:
                if remove_gaps:
                    are_neighbours = ((MSA[b1] != 20) * (MSA[b2] != 20) * (MSA[b1] != MSA[b2])).sum() / (
                                (MSA[b1] != 20) * (MSA[b2] != 20)).sum() < threshold
                else:
                    are_neighbours = (MSA[b1] != MSA[b2]).mean() < threshold
                num_neighbours[b1] += are_neighbours
                num_neighbours[b2] += are_neighbours
    return np.asarray(num_neighbours, dtype=curr_int)


def get_focusing_weights(all_sequences, all_weights, WT, targets_Beff, step=0.5):
    homology = 1 - (all_sequences == all_sequences[WT]).mean(-1)
    Beff = all_weights.sum()
    targets_Beff = np.array(targets_Beff)
    Beff_min = targets_Beff.min()
    all_focusing_weights = np.ones(
        [len(all_weights), len(targets_Beff)], dtype=np.float32)

    if Beff_min < Beff:  # Attempt focusing.
        # First, determine the largest focusing coefficient to be applied.
        Beff_current = Beff
        focusing = 0.0
        all_focusings = [0.0]
        all_Beff = [Beff]
        while Beff_current > Beff_min:
            focusing += step
            focusing_weights = np.exp(-focusing * homology)
            Beff_current = (all_weights * focusing_weights).sum()
            all_focusings.append(focusing)
            all_Beff.append(Beff_current)
        # Next interpolate to determine the correct focusing coefficients to be applied to each target.
        f = interp1d(all_Beff, all_focusings, bounds_error=False)
        target_focusings = f(targets_Beff)
        target_focusings[targets_Beff > Beff] = 0.
        for l, target_focusing in enumerate(target_focusings):
            all_focusing_weights[:, l] = np.exp(-target_focusing * homology)
    return all_focusing_weights


def conservation_score(PWM, Beff, Bvirtual=5):
    eps = Bvirtual / (Bvirtual + Beff * (1 - PWM[:, -1]))
    PWM = PWM[:, :-1].copy()
    PWM /= PWM.sum(-1)[:, np.newaxis]
    PWM = eps[:, np.newaxis] / 20 + (1 - eps[:, np.newaxis]) * PWM
    conservation = np.log(20) - (- np.log(PWM) * PWM).sum(-1)
    return conservation


def compute_PWM(location, gap_threshold=0.3,
                neighbours_threshold=0.1, Beff=[10, np.inf], WT=0, scaled=False
                , return_list=True):
    '''
    Default call:
    pwm_utils.compute_PWM(absolute_path_to_a3m_alignment)
    4 outputs:
    1: Position Weight Matrix (L X 21) focused on the wild type (for the weighting network).
    2: Position Weight Matrix (L X 21) averaged across the entire alignment (for the coevolutionary network).
    3: Conservation Score (L) of the alignment (for the weighting network).
    4: Effective number of sequences within alignment Beff (for the weighting network).
    Requirements:
    numpy, pandas, numba, scipy.
    '''
    if not isinstance(Beff, list):
        Beff = [Beff]

    nBeff = len(Beff)

    all_sequences, all_labels = load_FASTA(
        location, remove_insertions=True, with_labels=True, drop_duplicates=True)
    sequences_with_few_gaps = (all_sequences == 20).mean(-1) < gap_threshold
    all_sequences = all_sequences[sequences_with_few_gaps]
    all_labels = all_labels[sequences_with_few_gaps]
    all_weights = 1.0 / \
                  count_neighbours(all_sequences, threshold=neighbours_threshold)

    ambiguous_residues = np.nonzero(all_sequences == 21)
    if len(ambiguous_residues[0]) > 0:
        all_sequences[ambiguous_residues[0], ambiguous_residues[1]] = 20
        PWM = weighted_average(all_sequences, all_weights.astype(curr_float), 21)
        consensus = np.argmax(PWM, axis=-1)
        all_sequences[ambiguous_residues[0], ambiguous_residues[1]] = consensus[ambiguous_residues[1]]

    all_focusing_weights = get_focusing_weights(
        all_sequences, all_weights, WT, Beff)
    all_weights_focused = all_weights[:, np.newaxis] * all_focusing_weights
    all_weights_focused /= all_weights_focused.mean(0)

    PWM = np.zeros([all_sequences.shape[-1], 21, nBeff], dtype=curr_float)
    for n in range(nBeff):
        PWM[:, :, n] = weighted_average(
            all_sequences, all_weights_focused[:, n].astype(curr_float), 21)
    if scaled:
        Beff = all_weights.sum()
        for n in range(nBeff):
            conservation = conservation_score(PWM[:, :, n], Beff, Bvirtual=5)
            PWM[:, :, n] *= conservation[:, np.newaxis]
    if nBeff == 1:
        PWM = PWM[:, :, 0]
    if return_list:
        outputs = {}
        if nBeff>1:
            outputs['PWM'] = [PWM[:,:,k] for k in range(nBeff)]
        else:
            outputs['PWM'] = PWM
        Beff_alignment = all_weights.sum()
        conservation = conservation_score(PWM[:,:,-1], Beff_alignment, Bvirtual=5)
        outputs['conservation'] = conservation
        outputs['Beff_alignment'] = Beff_alignment
        return outputs
    else:
        return PWM
