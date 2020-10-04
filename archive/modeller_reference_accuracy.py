import logging
from argparse import ArgumentParser

from .analysis import modeller_accuracy, reference_accuracy

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = ArgumentParser(description="modeller accuracy")

    parser.add_argument('-n',
                        '--name',
                        type=str)

    parser.add_argument('-v',
                        '--version',
                        type=int)

    return parser.parse_args()


def main():
    args = parse_args()
    modeller_accuracy('pfam', args.name, args.version)
    reference_accuracy('pfam', args.name, args.version)


if __name__ == '__main__':
    main()
