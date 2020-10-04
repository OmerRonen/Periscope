import logging
import os
from argparse import ArgumentParser

from .utils_old import yaml_load, yaml_save
from .globals import periscope_path

LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser(description="Mutating a model")
    parser.add_argument('-o', '--original_model', type=str)
    parser.add_argument('-m', '--new_model', type=str)
    parser.add_argument('-f', '--force', action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    original_model = args.original_model
    new_model = args.new_model
    force = args.force

    params = yaml_load(os.path.join(periscope_path, 'models', original_model, 'params.yml'))
    model_path = os.path.join(periscope_path, 'models', new_model)
    params['train']['path'] = model_path
    if os.path.exists(model_path) and not force:
        LOGGER.info(f'{new_model} already exist')
        return
    LOGGER.info(f'Creating folder {model_path}')
    os.mkdir(model_path)
    yaml_save(filename=os.path.join(params['train']['path'], 'params.yml'), data=params)


if __name__ == "__main__":
    main()
