from argparse import ArgumentParser

from ..utils.tm import model_modeller_tm_scores


def parse_args():
    parser = ArgumentParser(description="Calculate model tm scores vs modeller and upload to Drive")
    parser.add_argument('-n', '--name', type=str, help='model name')
    parser.add_argument('-t', '--target', type=str, help='target name', default=None)
    parser.add_argument('-f', '--fast', action='store_true')
    parser.add_argument('-s', '--sswt', type=str, help='secondary structure weight', default='5')
    parser.add_argument('-r', '--selectrr', type=str, help='number of restraints', default='all')

    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.name
    target = args.target
    fast = args.fast

    rr = str(args.selectrr)

    selectrr = f'{rr}.0L' if rr.isdigit() else rr
    sswt = args.sswt
    model_modeller_tm_scores(model_name, target, fast, selectrr=selectrr, sswt=sswt)


if __name__ == '__main__':
    main()
