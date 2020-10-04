import os
import logging

import pandas as pd
import numpy as np

from .analyzer import calculate_accuracy
from ..utils.constants import DATASETS, DATASETS_FULL, PATHS, yaml_load

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def _read_csv(filename):
    return pd.read_csv(filename, index_col=0).values


def _dataset_pre_post_filter(dataset_name):
    pre = len(getattr(DATASETS_FULL, dataset_name))
    post = len(getattr(DATASETS, dataset_name))
    return {'pre': pre, 'post': post}


def get_datasets_pre_post():
    ds = {}
    for d in 'train eval pfam cameo membrane cameo41'.split(' '):
        ds[d] = _dataset_pre_post_filter(d)
    return ds


# def evaluate_model(dataset, model_name):
#     tops = [10, 5, 2, 1]
#     categories = {
#         'S': {n: None
#               for n in tops},
#         'M': {n: None
#               for n in tops},
#         'L': {n: None
#               for n in tops}
#     }
#     for category in categories:
#         for top in tops:
#             acc = get_average_accuracy([dataset], category, top, model_name)
#             categories[category][top] = acc
#     return categories


def get_average_accuracy(datasets, category, top, model_name):
    acc_vec = []
    for d in datasets:
        d_path = os.path.join(PATHS.drive, model_name, 'predictions', d)
        for t in os.listdir(d_path):
            if t not in getattr(DATASETS_FULL, d):
                continue
            logits = _read_csv(os.path.join(d_path, t, 'prediction.csv'))
            gt = _read_csv(os.path.join(d_path, t, 'gt.csv'))
            acc = calculate_accuracy(logits, gt)[category][top]
            acc_vec.append(acc)
    print(f'Number of predictions is {len(acc_vec)}')
    return np.round(np.mean(acc_vec), 2)


def get_tm_stats(datasets, model_name):
    folds = []
    for d in datasets:
        d_path = os.path.join(PATHS.drive, model_name, 'tm_5_2.0L', d)
        for t in os.listdir(d_path):
            if t.split('.')[0] not in getattr(DATASETS_FULL, d):
                continue
            tm_data = yaml_load(os.path.join(d_path, t))[model_name]
            if tm_data is None:
                continue
            folds.append(int(tm_data > 0.5))

    return np.round(np.mean(folds), 2)
