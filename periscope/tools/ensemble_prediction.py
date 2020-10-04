import logging
import os
import shutil
from argparse import ArgumentParser

from ..analysis.analyzer import write_model_analysis
from ..utils.constants import PATHS
from ..utils.drive import upload_folder
from ..utils.utils import check_path, yaml_save

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser(description="Ensemble prediction averages multiple models")
    parser.add_argument('-n', '--model-name', type=str)
    parser.add_argument('-m', '--models', nargs='+')

    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model_name
    models = ['ccmpred_ms_2' ,'ccmpred_ms_3', 'ccmpred_ms_4', 'ccmpred_ms_5']
    LOGGER.info(f'Ensemble model for {models}')
    model_path = os.path.join(PATHS.models, model_name)
    check_path(model_path)
    yaml_save(os.path.join(model_path, 'models.yaml'), {'models': models})
    write_model_analysis(model=None, model_path=model_path, models=models, model_name=model_name)
    upload_folder(model_path, model_path.split('Periscope/')[-1])
    # no need to save the artifacts here
    shutil.rmtree(model_path)


if __name__ == "__main__":
    main()
