import logging

from .data_handler import ProteinDataHandler
from .globals import DATASETS

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger()


def main():
    for target in DATASETS['eval']:
        dh = ProteinDataHandler(target, structures_version=3)

        nas =(dh.target_pdb_cm==-1).sum()
        LOGGER.info(f'Nas: {nas}')


if __name__ == '__main__':
    main()