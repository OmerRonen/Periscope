from argparse import ArgumentParser

from ..utils.constants import DATASETS
from ..data.creator import DataCreator


def parse_args():
    parser = ArgumentParser(description="Generates modeller data")
    parser.add_argument('-d', '--dataset', type=str)

    return parser.parse_args()


def main():
    args = parse_args()
    dataset = getattr(DATASETS, args.dataset)
    for target in dataset:
        DataCreator(target, 1).get_average_modeller_dm(4)


if __name__ == '__main__':
    main()


