import logging
import os
import shutil

from argparse import ArgumentParser
import tensorflow as tf

from .analysis import write_model_analysis
from .cm_predictor import ContactMapPredictor
from .drive import upload_folder
from .globals import MODELS_PATH, periscope_path
from .protein_net import NetParams

if os.path.exists('models/cnn_sanity_check'):
    shutil.rmtree('models/cnn_sanity_check')

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger = tf.get_logger()

logger.propagate = False


def parse_args():
    parser = ArgumentParser(description="Upload local folder to Google Drive")
    parser.add_argument('-n', '--name', type=str, help='model name')
    parser.add_argument('-t', '--train', action="store_true")
    parser.add_argument('-i', '--info', action="store_true")
    parser.add_argument('-u', '--upload', action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.name
    info = args.info
    train = args.train
    upload = args.upload

    params = NetParams(os.path.join(periscope_path,
                                    'slurm_scripts/params.yml')).params

    if model_name in os.listdir(os.path.join(periscope_path, 'models')):
        params = NetParams(
            os.path.join(periscope_path, 'models', model_name,
                         'params.yml')).params
    params['train']['path'] = os.path.join(MODELS_PATH, model_name)

    cm_pred = ContactMapPredictor(params)
    if train:
        cm_pred.train_and_evaluate()

    if info:
        write_model_analysis(model_name)

    if upload:
        upload_folder(cm_pred.path, cm_pred.path.split('Periscope/')[-1])
        # no need to save the artifacts here
        shutil.rmtree(cm_pred.artifacts_path)


if __name__ == '__main__':
    main()
