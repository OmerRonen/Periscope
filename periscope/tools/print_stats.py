import logging
import os

import pandas as pd
from argparse import ArgumentParser

from periscope.utils.constants import PATHS
from ..analysis.analyzer import ds_accuracy
from ..net.contact_map import get_model_by_name
from ..analysis.stats import get_datasets_pre_post, get_average_accuracy, get_tm_stats

LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser(description="generates data for taget")
    parser.add_argument('model')

    return parser.parse_args()


def main():
    print(get_datasets_pre_post())
    model = parse_args().model
    # tm = get_tm_stats(['membrane', 'cameo41', 'cameo'], model)
    # LOGGER.info(f'{model} average tm is {tm}')
    if model != 'modeller':
        # average_acc = get_average_accuracy(['membrane', 'cameo41', 'cameo'], "L", 1, model)
        # LOGGER.info(f'Average accurage of top L long range contacts is {average_acc}')
        for d in ['membrane','cameo41', 'cameo']:
            accuracy = ds_accuracy(dataset=d, model=get_model_by_name(model))
            LOGGER.info(f'Accuracy for {d}:\n{accuracy}')
            accuracy.to_csv(os.path.join(PATHS.models, model, 'predictions', f'{d}.csv'))


if __name__ == '__main__':
    main()
