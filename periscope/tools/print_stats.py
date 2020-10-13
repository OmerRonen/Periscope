import logging
import os

import matplotlib.pyplot as plt
from argparse import ArgumentParser

from ..utils.constants import PATHS, yaml_load
from ..analysis.analyzer import ds_accuracy
from ..net.contact_map import get_model_by_name
from ..analysis.stats import get_datasets_pre_post

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
        for d in ['membrane', 'cameo41', 'cameo']:
            accuracy = ds_accuracy(dataset=d, model=get_model_by_name(model))
            LOGGER.info(f'Accuracy for {d}:\n{accuracy}')
            accuracy.to_csv(os.path.join(PATHS.models, model, 'predictions', f'{d}.csv'))

    msa_stats = yaml_load('/Users/omerronen/Google Drive (omerronen10@gmail.com)/Periscope/data/stats/stats_msa.yaml')
    msa_lengths = list(msa_stats['msa'].values())
    msa_tempaltes = list(msa_stats['tempaltes'].values())
    plt.hist(msa_tempaltes, bins=50)
    plt.title('Number of Templates')
    plt.savefig(os.path.join(PATHS.periscope, 'data', 'figures', 'templates.png'))
    plt.close()
    plt.hist(msa_lengths)
    plt.title('MSA Depth')
    plt.savefig(os.path.join(PATHS.periscope, 'data', 'figures', 'depths.png'))
    plt.close()



if __name__ == '__main__':
    main()
