import logging
import os
import random

import matplotlib.pylab as plt
from matplotlib.colors import to_rgb
from matplotlib import patches as mpatches
import numpy as np

from ..utils.constants import PATHS, DATASETS
from ..utils.utils import yaml_load, get_target_dataset

datasets = ['cameo', 'cameo41', 'membrane', 'pfam']

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def _get_scores(target):
    dataset = get_target_dataset(target)
    fname = f'/Users/omerronen/Google Drive (omerronen10@gmail.com)/Periscope/models/nips_model/tm_5_2.0L/{dataset}/{target}.yaml'
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

        our_scores = np.array([scores[s]['ms_2'] for s in proteins])
        modeller = np.array([scores[s]['modeller'] for s in proteins])
        refs_scores = np.array([scores[s]['ref'] for s in proteins])

        n_homs = np.array([scores[s]['n_homs'] for s in proteins])
        n_homs = n_homs / 1000  # / (np.max(n_homs) / 20)
        n_refs = np.array([float(scores[s]['n_refs']) for s in proteins])
        n_refs = n_refs  # / np.max(n_refs)

        modeller_correct_fold = np.sum(np.array(modeller) > 0.5)
        ms_correct_fold = np.sum(np.array(our_scores) > 0.5)

        LOGGER.info(
            f'{dataset} (n = {len(modeller)}):\nModeller correct folds: {modeller_correct_fold}\nMs correct fold {ms_correct_fold}')

        modeller_win = np.sum(np.logical_and(np.array(modeller) > 0.5, np.array(our_scores) < 0.5))
        ms_win = np.sum(np.logical_and(np.array(our_scores) > 0.5, np.array(modeller) < 0.5))

        LOGGER.info(f'{dataset} (n = {len(modeller)}):\nModeller win: {modeller_win}\nMs win {ms_win}')
        # color = np.array(['blue' if s>0.5 else 'red' for s in our_scores])
        tt = np.where(np.logical_and(modeller > 0.5, our_scores > 0.5))
        ft = np.where(np.logical_and(modeller < 0.5, our_scores > 0.5))
        tf = np.where(np.logical_and(modeller > 0.5, our_scores < 0.5))
        ff = np.where(np.logical_and(modeller < 0.5, our_scores < 0.5))
        our_scores -= refs_scores
        modeller -= refs_scores
        sc1 = plt.scatter(our_scores[tt], modeller[tt], c=n_refs[tt],
                    marker=10, label=dataset, s=n_homs[tt],
                    cmap=plt.cm.get_cmap('Greys'))
        sc = plt.scatter(our_scores[tt], modeller[tt], c=n_refs[tt],
                    marker=10, label=dataset, s=n_homs[tt],
                    cmap=plt.cm.get_cmap('Blues'))
        plt.scatter(our_scores[ft], modeller[ft], c=n_refs[ft],
                    marker=11, label=dataset, s=n_homs[ft],
                    cmap=plt.cm.get_cmap('Blues'))
        plt.scatter(our_scores[tf], modeller[tf], c=n_refs[tf],
                    marker=10, label=dataset, s=n_homs[tf],
                    cmap=plt.cm.get_cmap('Reds'))
        plt.scatter(our_scores[ff], modeller[ff], c=n_refs[ff],
                    marker=11, label=dataset, s=n_homs[ff],
                    cmap=plt.cm.get_cmap('Reds'))

        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        for i, s in enumerate(proteins):
            if True:  # dataset != 'membrane':
                dx = our_scores[i] - scores[s]['ref']
                dy = modeller[i] - scores[s]['ref']
                start = (scores[s]['ref'], dx)
                end = (scores[s]['ref'], dy)
                # plt.arrow(start[0], end[0], start[1], end[1], linestyle=(0, (1, 10)),
                #           color=color[i])

            # if our_scores[i] > 0.5 and modeller[i] > 0.5:
            #     continue
            # plt.annotate(s, (our_scores[i], modeller[i]), size=6)

        plt.plot([-1, 1], [-1, 1], '--', c='black', alpha=0.3)
        # plt.plot([0.5, 0.5], [0, 1], '--', c='black', alpha=0.3)
        # plt.plot([0, 1], [0.5, 0.5], '--', c='black', alpha=0.3)

        legend_elements = [plt.Line2D([0], [0], marker='o', color='blue', label='Periscope > 0.5', linestyle='None'),
                           plt.Line2D([0], [0], marker='o', color='red', label='Periscope < 0.5', linestyle='None'),
                           plt.Line2D([0], [0], marker=11, color='black', label='Modeller < 0.5', linestyle='None'),
                           plt.Line2D([0], [0], marker=10, color='black', label='Modeller > 0.5', linestyle='None')
                           ]

        plt.xlabel("Our Method - Ref")
        plt.ylabel("Modeller - Ref")
        cb = plt.colorbar(sc1)
        cb.ax.get_yaxis().labelpad = 25

        cb.ax.set_ylabel('# of templates', rotation=270)

        plt.legend(handles=[mpatches.Patch(color='black', label='Type1')])
        legend1 = plt.legend(legend_elements,
                             ["Periscope > 0.5", "Periscope < 0.5", "Modeller < 0.5", "Modeller > 0.5"], loc=2,
                             fontsize=6, fancybox=True
                             )
        plt.legend(*sc.legend_elements("sizes", num=6), loc=3, fancybox=True, title='MSA depth\n(Thousends)', fontsize=6)
        plt.gca().add_artist(legend1)
        # plt.legend(loc='upper left')
        plt.title(f'{dataset}')
        plt.savefig(os.path.join(PATHS.periscope, 'data', 'figures', f'tm_scores_{dataset}.png'))
        plt.close()


if __name__ == '__main__':
    main()
