from argparse import ArgumentParser
from ..data.creator import DataCreator
from ..net.contact_map import get_model_by_name
from ..utils.constants import DATASETS, N_REFS


def parse_args():
    parser = ArgumentParser(description="Generates modeller data")
    parser.add_argument('-d', '--dataset', type=str)
    parser.add_argument('-m', '--model_name', type=str)

    return parser.parse_args()


def main():
    args = parse_args()
    dataset = getattr(DATASETS, args.dataset)
    model = get_model_by_name(args.model_name)
    for target in dataset:
        try:
            DataCreator(target, N_REFS).run_modeller_templates(model=model)
        except FileNotFoundError:
            pass

if __name__ == '__main__':
    main()
