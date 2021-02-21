from argparse import ArgumentParser

from periscope.data.creator import DataCreator
from periscope.utils.constants import DATASETS


def parse_args():
    parser = ArgumentParser(description="generates raptor x secondary structure predictions")
    parser.add_argument('protein', type=str, help='target protein', default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    p = args.protein
    dc = DataCreator(p, train=False)
    if not dc.has_msa:
        dc._run_hhblits()
    _ = dc.generate_data()


if __name__ == '__main__':
    main()
