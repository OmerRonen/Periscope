import os

import matplotlib.pylab as plt

from argparse import ArgumentParser

from ..utils.constants import PATHS
from ..utils.protein import Protein


def parse_args():
    parser = ArgumentParser(description="Create plots")
    parser.add_argument('-p', '--protein', type=str)

    return parser.parse_args()


def main():
    args = parse_args()
    protein = args.protein
    cm = Protein(protein[0:4], protein[4]).cm
    plt.matshow(cm,origin='lower', cmap='Greys')
    plt.savefig(os.path.join(PATHS.periscope, 'data', f'{protein}_cm.png'))
    plt.close()


if __name__ == '__main__':
    main()
