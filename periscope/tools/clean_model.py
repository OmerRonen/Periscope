import os
from ..utils.constants import PATHS

from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description="Predict using RaptorX")
    parser.add_argument('model', type=str, help='model name', default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    model = args.model
    model_path = os.path.join(PATHS.models, model)
    files = os.listdir(model_path)
    for f in files:
        if f == 'params.yml' or f == 'artifacts':
            continue
        os.remove(os.path.join(model_path, f))


if __name__ == '__main__':
    main()
