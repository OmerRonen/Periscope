import logging
import os
import random
import sys

import numpy as np

from ..utils.utils import yaml_load, yaml_save
from ..data.creator import DataCreator
from ..utils.constants import DATASETS_FULL, N_REFS, PATHS

logging.basicConfig()
LOGGER = logging.getLogger(__name__)


def main(batch, n_batches, dataset_name):
    batch = int(batch)
    LOGGER.info('working on batch %s' % str(batch))
    LOGGER.warning('working on batch %s' % str(batch))
    dataset = getattr(DATASETS_FULL, dataset_name)
    batch_size = np.ceil(len(dataset) / n_batches)
    LOGGER.info('batch size is %s' % batch_size)
    errs_file = os.path.join(PATHS.data, 'pp_errs.yaml')

    i = 1
    n = len(dataset)
    batch_set = dataset[int(((batch - 1) / n_batches) *
                            n):int((batch / n_batches) * n)]
    random.shuffle(batch_set)
    for protein_chain in batch_set:
        LOGGER.info('processing protein %s number %s' % (protein_chain, i))
        try:
            dc = DataCreator(protein_chain, n_refs=N_REFS)
            if not dc.recreated:
                dc.generate_data()
        except Exception:
            errors = {"proteins": []} if not os.path.isfile(errs_file) else yaml_load(errs_file)
            errors['proteins'].append(protein_chain)
            yaml_save(data=errors, filename=errs_file)
        i += 1


if __name__ == '__main__':
    batch = sys.argv[1]
    LOGGER.info('batch number %s\n' % batch)

    n_bathces = int(sys.argv[2])
    LOGGER.info('number of batches %s\n' % n_bathces)

    dataset_name = sys.argv[3]
    LOGGER.info('dataset name %s\n' % dataset_name)

    main(batch, n_bathces, dataset_name)
