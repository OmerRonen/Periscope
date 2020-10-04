import logging
import random

from .data_handler import ProteinDataHandler
import numpy as np
from evcouplings.utils.system import ResourceError
import sys

from .globals import DATASETS

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def main(batch, n_batches, dataset_name):
    batch = int(batch)
    LOGGER.info('working on batch %s' % str(batch))
    dataset = DATASETS[dataset_name]
    batch_size = np.ceil(len(dataset) / n_batches)
    LOGGER.info('batch size is %s' % batch_size)

    i = 1
    n = len(dataset)
    batch_set = dataset[int(((batch - 1) / n_batches) *
                            n):int((batch / n_batches) * n)]
    random.shuffle(batch_set)
    for protein_chain in batch_set:
        LOGGER.info('processing protein %s number %s' % (protein_chain, i))
        try:
            LOGGER.info(f'Data for {protein_chain}')
            ProteinDataHandler(protein_chain, mode='get')
        except (IndexError, ResourceError, ValueError, FileNotFoundError) as e:
            LOGGER.warning('problem with protein %s\n\nError: %s' %
                           (protein_chain, e))

        i += 1


if __name__ == '__main__':
    batch = sys.argv[1]
    LOGGER.info('batch number %s\n' % batch)

    n_bathces = int(sys.argv[2])
    LOGGER.info('number of batches %s\n' % n_bathces)

    dataset_name = sys.argv[3]
    LOGGER.info('dataset name %s\n' % dataset_name)

    main(batch, n_bathces, dataset_name)
