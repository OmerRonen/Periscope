import os
import re
import subprocess
import tempfile

from argparse import ArgumentParser
import numpy as np
import Bio
from Bio import SeqIO

from periscope.data.creator import DataCreator
from periscope.utils.constants import PATHS
from periscope.utils.protein import Protein
from periscope.utils.utils import get_target_hhblits_path, check_path, get_target_ccmpred_file


def parse_args():
    parser = ArgumentParser(description="generate")
    parser.add_argument('proteins', nargs='+', default=[],
                        help='Folders to exclude')

    return parser.parse_args()


def get_target_path(target, family=None):
    if family is not None:
        fam_path = os.path.join(PATHS.periscope, 'data', 'families', family, target)
        check_path(fam_path)
        return fam_path

    f_name = target
    t_path = os.path.join(PATHS.proteins, target[1:3], f_name)
    check_path(t_path)
    return t_path


def get_target_hhblits_path(target):
    return os.path.join(get_target_path(target), 'hhblits_new')


def _get_seq(p):
    seq = Protein(p[0:4], p[4]).str_seq
    seq_record = SeqIO.SeqRecord(Bio.Seq.Seq(seq), name=p, id=p)
    # seq_record.seq = re.sub('[^GATC]', "", str(seq).upper())
    return seq_record


def _run_hhblits(proteins, name):
    # Generates multiple sequence alignment using hhblits

    sequences = [_get_seq(p) for p in proteins]
    target_hhblits_path = get_target_hhblits_path(name)
    check_path(target_hhblits_path)

    fname = os.path.join(target_hhblits_path, name + '_seed.fasta')

    # SeqIO.write(sequences, query, "fasta")
    run_clustalo(sequences, fname)
    output_hhblits = os.path.join(target_hhblits_path, name + '.a3m')
    output_reformat1 = os.path.join(target_hhblits_path, name + '.a2m')
    output_reformat2 = os.path.join(target_hhblits_path, name + '_.fasta')

    db_hh = '/vol/sci/bio/data/or.zuk/projects/ContactMaps/data/uniref30/UniRef30_2020_06'

    hhblits_params = '-n 3 -e 1e-3 -maxfilt 10000000000 -neffmax 20 -nodiff -realign_max 10000000000'

    hhblits_cmd = f'hhblits -i {fname} -d {db_hh} {hhblits_params} -oa3m {output_hhblits}'
    subprocess.run(hhblits_cmd, shell=True)
    # subprocess.run(hhblits_cmd, shell=True, stdout=open(os.devnull, 'wb'))
    reformat = ['reformat.pl', output_hhblits, output_reformat1]
    subprocess.run(reformat)

    reformat = ['reformat.pl', output_reformat1, output_reformat2]
    subprocess.run(reformat)
    fam_msa = "/vol/sci/bio/data/or.zuk/projects/ContactMaps/src/Periscope/data/families/xcl1_family/msa.fasta"
    os.rename(output_reformat2, fam_msa)


def run_clustalo(sequences, fname):
    # SeqIO.write(sequences, fname, 'fasta')
    # handle = open(fname, "w")
    SeqIO.write(sequences, fname, "fasta")
    # for seq in sequences:
    #     SeqIO.write(seq, handle, "fasta")

    cmd = f'clustalo -i {fname} -o {fname} --force'
    subprocess.run(cmd, shell=True)


proteins = ['2jp1B', '2n54A', '7jh1A', '1j9oA', '1j8iA']
name = 'xcl1_family'
# _run_hhblits(proteins, name)
dc = DataCreator("2n54A", family=name)
ccmpred = dc.ccmpred
msa = dc._parse_msa()
l = len(msa['1j9oA'])
ccmpred_full = np.zeros(shape=(l,l))
target_seq_arr = dc.target_seq_msa
inds = np.where(target_seq_arr != '-')[0]
row_idx = np.array(inds)
col_idx = np.array(inds)
ccmpred_full[row_idx[:, None], col_idx] = ccmpred
np.savetxt(X=ccmpred_full,fname=ccmpred_mat_file)
# self = dc.aligner