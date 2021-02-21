from argparse import ArgumentParser

from periscope.utils.constants import DATASETS


def parse_args():
    parser = ArgumentParser(description="generates raptor x secondary structure predictions")
    parser.add_argument('protein', type=str, help='target protein', default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    p = args.protein
    _ = get_raptor_ss(target=p)


if __name__ == '__main__':
    main()
