import logging
import os
import random

import matplotlib.pylab as plt
from matplotlib.colors import to_rgb
import numpy as np

from ..utils.constants import PATHS, DATASETS
from ..utils.utils import yaml_load, get_target_dataset

datasets = ['cameo', 'cameo41', 'membrane', 'pfam']

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def _get_scores(target):
    dataset = get_target_dataset(target)
    fname = f'/Users/omerronen/Google Drive (omerronen10@gmail.com)/Periscope/models/ms_2/tm_5_2.0L/{dataset}/{target}.yaml'
    if not os.path.isfile(fname):
        return
    scores = yaml_load(fname)
    if len(scores) != 5:
        return
    return scores


def main():
    markers = {'pfam': '8', 'cameo': 'p', 'cameo41': "*", 'membrane': "D"}
    for dataset in datasets:
        scores = {}
        ds = getattr(DATASETS, dataset)

        for target in ds:
            s = _get_scores(target)
            if s is None:
                continue
            scores[target] = s
        proteins = list(scores.keys())

        def _check_score(s):
            return scores[s]['modeller'] is not None

        proteins = [s for s in proteins if _check_score(s)]

        our_scores = [scores[s]['ms_2'] for s in proteins]
        modeller = [scores[s]['modeller'] for s in proteins]
        n_homs = np.array([scores[s]['n_homs'] for s in proteins])
        n_homs = n_homs / (np.max(n_homs) / 20)
        n_refs = np.array([float(scores[s]['n_refs']) for s in proteins])
        n_refs = n_refs / np.max(n_refs)

        modeller_correct_fold = np.sum(np.array(modeller) > 0.5)
        ms_correct_fold = np.sum(np.array(our_scores) > 0.5)

        LOGGER.info(
            f'{dataset} (n = {len(modeller)}):\nModeller correct folds: {modeller_correct_fold}\nMs correct fold {ms_correct_fold}')

        modeller_win = np.sum(np.logical_and(np.array(modeller) > 0.5, np.array(our_scores) < 0.5))
        ms_win = np.sum(np.logical_and(np.array(our_scores) > 0.5, np.array(modeller) < 0.5))

        LOGGER.info(f'{dataset} (n = {len(modeller)}):\nModeller win: {modeller_win}\nMs win {ms_win}')
        clr ='blue' #colors[dataset]
        r, g, b = to_rgb(clr)
        # r, g, b, _ = to_rgba(color)
        color = [(r, g, b, alpha) for alpha in n_refs]

        plt.scatter(our_scores, modeller, c=color, marker=markers[dataset], label=dataset, s=n_homs)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        for i, s in enumerate(proteins):
            if True:#dataset != 'membrane':
                dx = our_scores[i] - scores[s]['ref']
                dy = modeller[i] - scores[s]['ref']
                start = (scores[s]['ref'], dx)
                end = (scores[s]['ref'], dy)
                plt.arrow(start[0], end[0], start[1], end[1], linestyle=(0, (1, 10)),
                          color=color[i])

            # if our_scores[i] > 0.5 and modeller[i] > 0.5:
            #     continue
            # plt.annotate(s, (our_scores[i], modeller[i]), size=6)

        plt.plot([0, 1], [0, 1], '--', c='black', alpha=0.3)
        plt.plot([0.5, 0.5], [0, 1], '--', c='black', alpha=0.3)
        plt.plot([0, 1], [0.5, 0.5], '--', c='black', alpha=0.3)

        plt.xlabel("Our Method")
        plt.ylabel("Modeller")
        # plt.legend(loc='upper left')
        plt.title(f'TMscores - our method and modeller vs native structure {dataset}')
        plt.savefig(os.path.join(PATHS.periscope, 'data', 'figures', f'tm_scores_{dataset}.png'))
        plt.close()


if __name__ == '__main__':
    main()
