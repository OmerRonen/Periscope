from argparse import ArgumentParser

from ..data.creator import DataCreator


def parse_args():
    parser = ArgumentParser(description="Calculate model tm scores vs modeller and upload to Drive")
    parser.add_argument('-t', '--targets', nargs="+", help='target name')

    return parser.parse_args()


def main():
    args = parse_args()

    targets = args.targets
    for t in targets:
        dc = DataCreator(t)
        dc.scores


if __name__ == '__main__':
    main()
