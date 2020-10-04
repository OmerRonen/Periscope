import logging
import os

import numpy as np
import pandas as pd
import datetime as dt

from argparse import ArgumentParser

from periscope.data.creator import DataCreator
from periscope.utils.constants import DATASETS, PATHS
from periscope.utils.protein import Protein
from periscope.utils.utils import get_modeller_pdb_file, check_path

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser(description="updates the generic data")
    parser.add_argument('-d', '--dataset', type=str, help='data set name')

    return parser.parse_args()


def main():
    args = parse_args()
    dataset = args.dataset
    targets = getattr(DATASETS, dataset)

    for target in targets:
        ref_file = os.path.join(PATHS.data, dataset, 'reference', f'{target}.csv')
        dc = DataCreator(target)
        ref = dc.refs_contacts
        pd.DataFrame(ref).to_csv(ref_file)
        modeller = dc.modeller_cm
        modeller_file = os.path.join(PATHS.data, dataset, 'modeller', f'{target}.csv')
        pd.DataFrame(modeller).to_csv(modeller_file)
        #
        # if False:
        #
        #     check_path(os.path.join(PATHS.data, dataset, 'reference'))
        #     ref_file_creation = yesterday
        #     if os.path.isfile(ref_file):
        #         ref_file_creation = dt.datetime.fromtimestamp(os.path.getctime(ref_file)).date()
        #     modeller_file = os.path.join(PATHS.data, dataset, 'modeller', f'{target}.csv')
        #     check_path(os.path.join(PATHS.data, dataset, 'modeller'))
        #     modeller_file_creation = yesterday
        #     if os.path.isfile(modeller_file):
        #         modeller_file_creation = dt.datetime.fromtimestamp(os.path.getctime(ref_file)).date()
        #
        #     LOGGER.info(target)
        #
        #     if ref_file_creation != today:
        #         dms = DataCreator(target, 10).k_reference_dm_test
        #         if dms is None:
        #             continue
        #         dms[np.logical_or(dms == -1, dms == 0)] = np.nan
        #
        #         ref = np.array(np.nanmin(dms, 2) < 8, dtype=np.int)
        #         ref[np.isnan(ref)] = -1
        #         pd.DataFrame(ref).to_csv(ref_file)
        #     if modeller_file_creation != today:
        #         mode_file = get_modeller_pdb_file(target=target, templates=True, n_struc=1)
        #         if os.path.isfile(mode_file):
        #             modeller = Protein(target[0:4], target[4], pdb_path=PATHS.modeller).cm
        #             pd.DataFrame(modeller).to_csv(modeller_file)


if __name__ == '__main__':
    main()
