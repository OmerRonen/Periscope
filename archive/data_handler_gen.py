import logging
from argparse import ArgumentParser

from .data_handler import ProteinDataHandler
from .globals import DATASETS

logging.getLogger().setLevel(logging.INFO)

def parse_args():
    parser = ArgumentParser(description="Initializing data handler")
    parser.add_argument('-d', '--dataset', type=str)

    return parser.parse_args()


def main():
    args = parse_args()
    data = DATASETS[args.dataset]
    for protein in data:
        ProteinDataHandler(protein)


if __name__ == '__main__':
    main()
