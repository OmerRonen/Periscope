import logging
from argparse import ArgumentParser

from ..data.creator import DataCreator

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser(description="generate features for a given target")
    parser.add_argument('-p', '--protein', type=str)
    parser.add_argument('-f', '--features', nargs='+')

    return parser.parse_args()


def main():
    args = parse_args()
    dc = DataCreator(args.protein, 1)
    LOGGER.info(f'Generating features: {args.features}')
    for f in args.features:
        getattr(dc, f)

if __name__ == '__main__':
    main()