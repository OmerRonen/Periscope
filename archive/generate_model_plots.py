import logging
import os
from argparse import ArgumentParser

import pandas as pd
from .analysis import plot_target_analysis

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = ArgumentParser(description="Create plots")
    parser.add_argument('-n', '--model_name', type=str)
    parser.add_argument('-r', '--reference_model', type=str)

    return parser.parse_args()


def _get_logits(model_name, target):
    if model_name is None:
        return

    model_path = f'/Users/omerronen/Google Drive (omerronen10@gmail.com)/Periscope/models/{model_name}/artifacts'

    data_path = os.path.join(model_path, 'data', target)

    logits_file = os.path.join(data_path, 'predicted_logits.csv')
    if not os.path.isfile(logits_file):
        raise FileNotFoundError

    predicted_logits = pd.read_csv(logits_file, index_col=0).values
    return predicted_logits


if __name__ == "__main__":

    args = parse_args()
    model_name = args.model_name
    model_drive_path = f'/Users/omerronen/Google Drive (omerronen10@gmail.com)/Periscope/models/{model_name}/artifacts'

    for prot in os.listdir(os.path.join(model_drive_path, 'data')):

        data_path = os.path.join(model_drive_path, 'data', prot)
        try:

            predicted_logits = _get_logits(args.model_name, prot)
            ref_predicted_logits = _get_logits(args.reference_model, prot)

        except FileNotFoundError:
            continue

        plot_target_analysis(predicted_logits=predicted_logits,
                             predicted_logits2=ref_predicted_logits,
                             data_path=data_path,
                             figures_path=os.path.join(model_drive_path,
                                                       'figures'),
                             target=prot,
                             model_name=model_name,
                             ref_name=args.reference_model,
                             main_only=False)
