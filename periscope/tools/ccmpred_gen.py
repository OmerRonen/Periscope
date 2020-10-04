import logging
from argparse import ArgumentParser

from ..data.creator import DataCreator
from ..utils.constants import DATASETS

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = ArgumentParser(description="Generate ccmpred for dataset")

    parser.add_argument('-d', '--dataset', type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    for protein in getattr(DATASETS, args.dataset):
        dc = DataCreator(protein, 10)
        if dc.has_refs:
            dc.ccmpred


if __name__ == '__main__':
    main()
