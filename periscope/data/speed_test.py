import time
import logging
from argparse import ArgumentParser

import numpy as np

from periscope.data.creator import DataCreator
from periscope.utils.constants import FEATURES, DATASETS

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser(description="Check speed of data generator")

    parser.add_argument('dataset', type=str, help='dataset name', default=None)
    parser.add_argument('n', type=int, help='number of proteins', default=100)

    return parser.parse_args()


def test_protein(protein):
    data = {}

    start_time = time.time()
    data_creator = DataCreator(protein)
    end_time = time.time()
    LOGGER.info(f'creator takes  {end_time - start_time}')
    if not data_creator.has_refs:
        return

    # start_time = time.time()
    # evfold = np.expand_dims(data_seeker.evfold, axis=2)
    # end_time = time.time()
    # LOGGER.info(f'Evfold takes  {end_time-start_time}')
    start_time = time.time()
    ccmpred = np.expand_dims(data_creator.ccmpred, axis=2)
    data[FEATURES.ccmpred] = ccmpred
    end_time = time.time()
    LOGGER.info(f'CCmpred takes  {end_time - start_time}')

    start_time = time.time()

    data[FEATURES.k_reference_dm_conv] = data_creator.k_reference_dm_test
    end_time = time.time()
    LOGGER.info(f'k_reference_dm_test takes  {end_time - start_time}')
    start_time = time.time()

    data[FEATURES.seq_refs] = data_creator.seq_refs_ss_acc
    end_time = time.time()
    LOGGER.info(f'seq_refs_ss_acc takes  {end_time - start_time}')

    start_time = time.time()

    data[FEATURES.pwm_w] = data_creator.pwm_w
    data[FEATURES.pwm_evo] = data_creator.pwm_evo_ss
    data[FEATURES.conservation] = data_creator.conservation
    data[FEATURES.beff] = data_creator.beff
    end_time = time.time()
    LOGGER.info(f'scores takes  {end_time - start_time}')
    start_time = time.time()

    data[FEATURES.properties_target] = data_creator.raptor_properties
    end_time = time.time()
    LOGGER.info(f'raptor_properties takes  {end_time - start_time}')
    data[FEATURES.seq_target] = data_creator.seq_target


def main():
    args = parse_args()
    proteins = getattr(DATASETS, args.dataset)
    for p in proteins[0:args.n]:
        test_protein(p)


if __name__ == '__main__':
    main()
