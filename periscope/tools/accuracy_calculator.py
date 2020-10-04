import logging
import os
from argparse import ArgumentParser

from ..analysis.analyzer import modeller_accuracy, reference_accuracy
from ..utils.constants import PATHS
from ..utils.drive import upload_folder

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = ArgumentParser(description="Baselines accuracy calculator")

    parser.add_argument('-d',
                        '--dataset',
                        type=str)

    return parser.parse_args()


def main():
    args = parse_args()

    ds = args.dataset

    modeller_accuracy(ds)
    modeller_path = os.path.join(PATHS.models, f'modeller', ds)
    upload_folder(modeller_path, modeller_path.split('Periscope/')[-1])

    reference_accuracy(ds)
    reference_path = os.path.join(PATHS.models,  f'reference', ds)
    upload_folder(reference_path, reference_path.split('Periscope/')[-1])


if __name__ == '__main__':
    main()
