from argparse import ArgumentParser
from ..data.creator import DataCreator
from ..net.contact_map import get_model_by_name
from ..utils.constants import DATASETS, N_REFS, DATASETS_FULL


def parse_args():
    parser = ArgumentParser(description="Generates modeller data")
    parser.add_argument('-d', '--dataset', type=str)
    parser.add_argument('-m', '--model_name', type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    dataset = getattr(DATASETS_FULL, args.dataset)
    if args.model_name is not None:
        model = get_model_by_name(args.model_name)
    else:
        model = None
    for target in dataset:
        try:
            DataCreator(target, N_REFS).run_modeller_templates(model=model)
        except FileNotFoundError:
            pass


if __name__ == '__main__':
    main()
