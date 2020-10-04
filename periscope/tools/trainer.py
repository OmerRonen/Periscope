import logging
import os
import shutil

from argparse import ArgumentParser
import tensorflow as tf

from ..net.params import NetParams
from ..utils.drive import upload_folder
from ..analysis.analyzer import save_model_predictions, write_model_analysis
from ..net.contact_map import ContactMapEstimator
from ..utils.constants import PATHS

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger = tf.get_logger()

logger.propagate = False

periscope_path = PATHS.periscope


def parse_args():
    parser = ArgumentParser(description="Upload local folder to Google Drive")
    parser.add_argument('-n', '--name', type=str, help='model name')
    parser.add_argument('-t', '--train', action="store_true")
    parser.add_argument('-i', '--info', action="store_true")
    parser.add_argument('-e', '--eval-dataset', type=str)
    parser.add_argument('-u', '--upload', action="store_true")
    parser.add_argument('-p', '--plot', action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.name
    info = args.info
    train = args.train
    upload = args.upload
    eval_dataset = args.eval_dataset

    params = NetParams(os.path.join(periscope_path,
                                    'slurm_scripts/params.yml')).params

    if model_name in os.listdir(os.path.join(periscope_path, 'models')):
        params = NetParams(
            os.path.join(periscope_path, 'models', model_name,
                         'params.yml')).params
    params['train']['path'] = os.path.join(PATHS.models, model_name)

    if train:
        model = ContactMapEstimator(params)

        model.train_and_evaluate()

    if info:
        params['train']['test_dataset'] = eval_dataset

        model = ContactMapEstimator(params)
        save_model_predictions(model)
        # write_model_analysis(model=model, model_path=os.path.join(model.artifacts_path, eval_dataset),
        #                      model_name=model_name, dataset=eval_dataset, plot=args.plot)

    if upload:
        path_to_upload = model.path if train else os.path.join(model.artifacts_path, eval_dataset)
        predicted_proteins = os.path.join(model.artifacts_path, f'predicted_proteins_{eval_dataset}.pkl')
        if info and os.path.isfile(predicted_proteins):
            os.remove(predicted_proteins)
        upload_folder(path_to_upload, path_to_upload.split('Periscope/')[-1])
        # no need to save the artifacts here
        # shutil.rmtree(os.path.join(model.artifacts_path, eval_dataset))


if __name__ == '__main__':
    main()
