import os

from argparse import ArgumentParser

from ..utils.drive import upload_folder
from ..utils.constants import PATHS
from ..utils.utils import yaml_save
from ..net.contact_map import get_model_by_name


def parse_args():
    parser = ArgumentParser(description="Number of parameters for each model")
    parser.add_argument('models', nargs="+", help='model names', default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    models = args.models
    n_params = {m: get_model_by_name(m).n_params for m in models}
    yaml_save(data=n_params, filename=os.path.join(PATHS.data, 'params_stat', 'params.yml'))
    upload_folder(os.path.join(PATHS.data, 'params_stat'), "params_stat")


if __name__ == '__main__':
    main()
