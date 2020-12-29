import logging
import os

import numpy as np
import pandas as pd

from argparse import ArgumentParser

from periscope.analysis.analyzer import get_model_predictions
from periscope.net.contact_map import get_model_by_name
from periscope.utils.constants import PATHS
from periscope.utils.drive import upload_folder
from periscope.utils.protein import Protein
from periscope.utils.utils import check_path, pkl_save

logging.getLogger().setLevel(logging.CRITICAL)


def parse_args():
    parser = ArgumentParser(description="Upload local folder to Google Drive")
    parser.add_argument('model', type=str, help='model name')
    parser.add_argument('proteins', nargs="+", help='target names')
    parser.add_argument('-o', "--outfolder", type=str, help='folder to save predictions')

    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model
    proteins = args.proteins
    outfolder = args.outfolder
    outfolder_full = os.path.join(PATHS.periscope, outfolder)
    check_path(outfolder_full)
    model = get_model_by_name(model_name)
    predictions = get_model_predictions(model, proteins=proteins, family='trypsin')
    logits = predictions['logits']
    weights = predictions['weights']

    for protein in proteins:
        outfolder_p = os.path.join(outfolder_full, protein)
        check_path(outfolder_p)
        sequence = list(Protein(protein[0:4], protein[4]).str_seq)
        logits_df = pd.DataFrame(np.squeeze(logits[protein]), columns=sequence, index=sequence)
        logits_df.to_csv(os.path.join(outfolder_p, 'logits.csv'))
        pkl_save(os.path.join(outfolder_p, 'weights.pkl'), weights[protein])

    upload_folder(outfolder_full, os.path.join(outfolder,protein))


if __name__ == '__main__':
    main()
