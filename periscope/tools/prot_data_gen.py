from argparse import ArgumentParser

from ..utils.constants import N_REFS
from ..data.creator import *

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser(description="generates data for taget")
    parser.add_argument('protein')

    return parser.parse_args()


def main():
    args = parse_args()
    target = args.protein
    dc = DataCreator(target, N_REFS)
    dc.generate_data()
    LOGGER.info(dc.metadata)


if __name__ == '__main__':
    main()
