import logging
from argparse import ArgumentParser

from ..phylo.asr import find_ancestors

logging.getLogger().setLevel(logging.CRITICAL)


def parse_args():
    parser = ArgumentParser(description="Reconstruct evolutionary tree")
    parser.add_argument('protein', type=str, help='target name')
    parser.add_argument('-n', '--n_seqs', type=int, default=None, help='maximal number of sequences')

    return parser.parse_args()


def main():
    args = parse_args()
    protein = args.protein
    n_seqs = args.n_seqs

    find_ancestors(protein, n_seqs)


if __name__ == '__main__':
    main()
