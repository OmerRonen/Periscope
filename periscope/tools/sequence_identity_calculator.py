import os
import logging
import subprocess
import tempfile

import numpy as np

from Bio import pairwise2
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from ..utils.constants import DATASETS, PATHS
from ..utils.protein import Protein
from ..utils.utils import yaml_save, get_target_dataset, write_fasta, read_fasta

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

test_proteins = set(DATASETS.membrane) | set(DATASETS.cameo) | set(DATASETS.cameo41) | set(DATASETS.pfam)
train_eval_proteins = set(DATASETS.train) | set(DATASETS.eval)

sequences = {t: Protein(t[0:4], t[4]).str_seq for t in test_proteins}


def calculate_sequence_identity_2(seq1, seq2):
    s1 = SeqRecord(Seq(seq1), id='s1', description='')
    s2 = SeqRecord(Seq(seq2), id='s2', description='')

    sequences = [s1, s2]
    inp = tempfile.NamedTemporaryFile()
    out = tempfile.NamedTemporaryFile()
    write_fasta(sequences, filename=inp.name)
    cmd = f'clustalo -i {inp.name} -o {out.name} --force'
    subprocess.run(cmd, shell=True)
    aln = read_fasta(out.name, full=True)

    return np.sum(np.array(aln[0].seq) == np.array(aln[1].seq)) / np.min([len(seq2), len(seq1)])


def calculate_sequence_identity(seq1, seq2):
    # match_score = 1
    # mismatch_score = 0
    # gap_open_score = -5
    # gap_extend_score = 0
    # alignment = pairwise2.align.globalms(seq1, seq2, match_score, mismatch_score, gap_open_score, gap_extend_score,
    #                                      one_alignment_only=True)[0]
    alignment = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True)[0]
    aligned_seq1 = np.array(list(alignment[0]))
    aligned_seq2 = np.array(list(alignment[1]))
    # aligned_seq2[aligned_seq2 == '-'] = '_'
    # return np.sum(aligned_seq1 == aligned_seq2) / np.min([len(seq1), len(seq2)])
    return np.mean(aligned_seq1 == aligned_seq2)


def main():
    violations = {}
    proteins_to_remove = []
    for train_protein in train_eval_proteins:
        seq1 = Protein(train_protein[0:4], train_protein[4]).str_seq
        LOGGER.info(train_protein)
        for test_protein in test_proteins:
            seq2 = sequences[test_protein]
            id = calculate_sequence_identity_2(seq1, seq2)
            if id > 0.25:
                LOGGER.info(
                    f'Violation of sequence identity {id} for {train_protein} - {test_protein}, {get_target_dataset(test_protein)}')
                violations[train_protein] = test_protein
                proteins_to_remove.append(train_protein)
                yaml_save(data=violations, filename=os.path.join(PATHS.periscope, 'data', 'id_violations.yaml'))
                break

    proteins_to_remove = set(proteins_to_remove)
    new_train = set(DATASETS.train).difference(proteins_to_remove)
    new_eval = set(DATASETS.eval).difference(proteins_to_remove)
    yaml_save(filename=os.path.join(PATHS.periscope, 'data', 'train_valid_4.yaml'), data={'proteins': list(new_train)})
    yaml_save(filename=os.path.join(PATHS.periscope, 'data', 'eval_valid_4.yaml'), data={'proteins': list(new_eval)})


if __name__ == "__main__":
    main()
