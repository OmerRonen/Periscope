import logging

from argparse import ArgumentParser

from ..data.creator import DataCreator

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser(description="generates data for protein/s")
    parser.add_argument('proteins', nargs='+', default=[],
                        help='Folders to exclude')

    return parser.parse_args()


def main():
    args = parse_args()
    proteins = args.proteins
    for target in proteins:
        DataCreator(target).generate_data()


if __name__ == '__main__':
    main()
