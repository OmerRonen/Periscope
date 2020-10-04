import argparse
import os
import logging

from periscope.utils.utils import check_path
from ..utils.constants import PATHS
from ..utils.drive import upload_folder
from ..utils.tm import save_modeller_scores

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate tm scores for modeller with many templates")

    parser.add_argument('dataset', type=str)

    return parser.parse_args()


def main():
    args = parse_args()
    dataset = args.dataset
    LOGGER.info(f'Working on {dataset}')
    path_to_upload = os.path.join(PATHS.models, 'modeller', dataset)
    check_path(path_to_upload)
    save_modeller_scores(dataset)

    upload_folder(path_to_upload, path_to_upload.split('Periscope/')[-1])


if __name__ == '__main__':
    main()
