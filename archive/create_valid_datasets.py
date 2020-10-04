import logging
import os
from argparse import ArgumentParser

from .data_handler import ProteinDataHandler
from .globals import periscope_path, csv_to_list
from .utils_old import yaml_save

LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser(description="Saves the valid data set")
    parser.add_argument('-d', '--dataset', type=str)

    return parser.parse_args()


def main():
    valid_proteins = []

    args = parse_args()

    dataset = args.dataset

    LOGGER.info('Creating valid dataset: %s' % dataset)

    lengths = {}

    ds = {
        'eval': "validation_proteins.csv",
        'train': 'training_proteins.csv',
        'pfam': '150pfam.csv'
    }

    proteins = csv_to_list(os.path.join(periscope_path, 'data', ds[dataset]))

    for protein in proteins:
        try:
            dh = ProteinDataHandler(protein, k=5)

            if len(dh.known_structures) > 1:
                valid_proteins.append(protein)

                l = (len(dh.protein.sequence) // 50) * 50 + 50
                if l in lengths:
                    lengths[l].append(protein)
                else:
                    lengths[l] = [protein]
        except Exception:
            LOGGER.info('Problem with %s' % protein)
            continue

    yaml_save(filename=os.path.join(periscope_path,
                                    'data/%s_valid.yaml' % dataset),
              data={'proteins': valid_proteins})
    yaml_save(filename=os.path.join(periscope_path,
                                    'data/%s_valid_batch.yaml' % dataset),
              data=lengths)


if __name__ == '__main__':
    main()
