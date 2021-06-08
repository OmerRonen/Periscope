import logging
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from ..data.creator import DataCreator
from ..net.contact_map import get_model_by_name
from ..utils.utils import get_target_dataset

logging.getLogger().setLevel(logging.CRITICAL)


def parse_args():
    parser = ArgumentParser(description="Upload local folder to Google Drive")
    parser.add_argument('model', type=str, help='model name')
    parser.add_argument('protein', type=str, help='target name')
    parser.add_argument('-f', "--family", type=str, help='protein family (optional)', default=None)
    parser.add_argument('-o', "--outfile", type=str, help='file to save prediction')
    parser.add_argument('-g', "--generate_data", action="store_true", help='if true generate the data')

    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model
    protein = args.protein
    outfile = args.outfile
    family = args.family
    model = get_model_by_name(model_name)
    if args.generate_data:
        _ = DataCreator(target=protein, family=family, train=False).generate_data()
    predictions = model.predict(proteins=[protein], family=family, dataset=get_target_dataset(protein))['logits']
    sequence = list(DataCreator(target=protein, family=family).str_seq)
    pd.DataFrame(np.squeeze(predictions[protein]), columns=sequence, index=sequence).to_csv(outfile)


if __name__ == '__main__':
    main()
