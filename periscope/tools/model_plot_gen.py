import logging
import os
from argparse import ArgumentParser

import pandas as pd

from ..utils.utils import get_target_dataset
from ..analysis.artist import make_art
from ..utils.constants import DATASETS, PATHS

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = ArgumentParser(description="Create plots")
    parser.add_argument('model_name', type=str)
    parser.add_argument('-r', '--reference_model', type=str)
    parser.add_argument('-d', '--dataset', type=str, default=None)
    parser.add_argument('-p', '--proteins', nargs='+', default=None)

    return parser.parse_args()


def _get_logits(model_name, target, dataset):
    if model_name is None:
        return

    model_path = f'/Users/omerronen/Google Drive (omerronen10@gmail.com)/Periscope/models/{model_name}/artifacts/{dataset}'

    data_path = os.path.join(model_path, 'data')

    logits_file = os.path.join(data_path, f'{target}_logits.csv')
    logits_file_old = os.path.join(data_path, target, f'predicted_logits.csv')

    logits_file = logits_file if os.path.isfile(logits_file) else logits_file_old
    if not os.path.isfile(logits_file):
        raise FileNotFoundError

    predicted_logits = pd.read_csv(logits_file, index_col=0).values
    return predicted_logits


if __name__ == "__main__":

    args = parse_args()
    model_name = args.model_name
    dataset = args.dataset
    proteins = args.proteins
    ds_given = dataset is not None

    proteins = getattr(DATASETS, dataset) if ds_given else proteins

    for protein in proteins:
        model_drive_path = os.path.join(PATHS.drive, f'{model_name}/predictions/{get_target_dataset(protein)}')

        if not os.path.exists(os.path.join(model_drive_path, protein)):
            continue
        # if len(os.listdir(os.path.join(model_drive_path, protein)))<3:
        #     continue
        make_art(model_name, protein)
        # if not ds_given:
        #     dataset = get_target_dataset(protein)
        #     model_drive_path = f'/Users/omerronen/Google Drive (omerronen10@gmail.com)/Periscope/models/{model_name}/artifacts/{dataset}'
        # try:
        #
        #     predicted_logits = _get_logits(args.model_name, protein, dataset)
        #     ref_predicted_logits = _get_logits(args.reference_model, protein, dataset)
        #
        # except FileNotFoundError:
        #     continue
        #
        # plot_target_analysis(predicted_logits=predicted_logits,
        #                      predicted_logits2=ref_predicted_logits,
        #                      figures_path=os.path.join(model_drive_path,
        #                                                'figures'),
        #                      target=protein,
        #                      model_name='our method',
        #                      ref_name='ConvNet',
        #                      main_only=args.full,
        #                      dataset=dataset)
