import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from argparse import ArgumentParser

from matplotlib.lines import Line2D
from scipy import interpolate

from ..utils.utils import get_data, get_raptor_logits
from ..utils.constants import PATHS, yaml_load, DATASETS
from ..analysis.analyzer import ds_accuracy, calc_accuracy
from ..net.contact_map import get_model_by_name
from ..analysis.stats import get_datasets_pre_post

LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser(description="generates data for taget")
    parser.add_argument('model')
    parser.add_argument('-d', '--datasets', nargs='+', default=None)

    return parser.parse_args()


def _plot_acc_vs_msa(dataset, model_name):
    targets = getattr(DATASETS, dataset)
    plot_data = {"Periscope": [], "RaptorX": [], "beff": [], "# of Templates": []}
    for t in targets:
        data = get_data(model_name, t)
        if data is None:
            continue
        raptor_logits = get_raptor_logits(t)
        logits = data['prediction']
        gt = data['gt']
        acc = calc_accuracy(pred=logits, gt=gt, top=2)
        acc_raptor = calc_accuracy(pred=raptor_logits, gt=gt, top=2)

        plot_data['Periscope'].append(acc)
        plot_data['RaptorX'].append(acc_raptor)
        plot_data['beff'].append(np.log(float(data['beff'])))
        templates = data['templates']
        n_refs = np.sum(np.max(np.max(templates, axis=0), axis=0) > 0)
        plot_data['# of Templates'].append(n_refs)

    plot_df = pd.DataFrame(plot_data)

    def _get_smooth_df(x_feature, df, thres=1000):
        df = df.sort_values(x_feature)
        if len(df[x_feature]) < thres:
            return df
        df[x_feature] += np.random.normal(0, 0.001, len(df[x_feature]))
        x = df[x_feature]
        x_new = np.linspace(np.min(x), np.max(x), len(x))
        Periscope_spline = interpolate.interp1d(x, df.Periscope)
        RaptorX_spline = interpolate.interp1d(x, df.RaptorX)
        periscope_smooth = Periscope_spline(x_new)
        raptorx_smooth = RaptorX_spline(x_new)
        plot_df_smooth = pd.DataFrame({x_feature: x_new, 'Periscope': periscope_smooth, 'RaptorX': raptorx_smooth})
        return plot_df_smooth

    plt.figure(figsize=(10, 8))
    plt.subplot(211)
    plot_df_smooth = _get_smooth_df('beff', plot_df)
    plt.plot('beff', 'Periscope', data=plot_df_smooth, marker='o', alpha=0.4, color='blue')
    plt.plot('beff', 'RaptorX', data=plot_df_smooth, marker='o', alpha=0.4, color="orange")
    plt.xlabel('$\log(Beff)$')
    plt.ylabel('Accuracy')
    custom_lines = [Line2D([0], [0], color='blue', lw=4),
                    Line2D([0], [0], color='orange', lw=4)]
    plt.legend(custom_lines, ['Periscope', 'RaptorX'], loc=2, prop={'size': 8})
    plot_df_smooth = _get_smooth_df('# of Templates', plot_df)

    plt.subplot(212)
    plt.plot('# of Templates', 'Periscope', data=plot_df_smooth, marker='o', alpha=0.4, color='blue')
    plt.plot('# of Templates', 'RaptorX', data=plot_df_smooth, marker='o', alpha=0.4, color="orange")
    plt.ylabel('Accuracy')
    plt.xlabel('# of Templates')
    plt.savefig(os.path.join(PATHS.models, model_name, 'predictions', f'{dataset}_msa_accuracy.png'))


def main():
    print(get_datasets_pre_post())
    args = parse_args()
    model = args.model
    datasets = args.datasets
    # tm = get_tm_stats(['membrane', 'cameo41', 'cameo'], model)
    # LOGGER.info(f'{model} average tm is {tm}')
    if model != 'modeller':
        # average_acc = get_average_accuracy(['membrane', 'cameo41', 'cameo'], "L", 1, model)
        # LOGGER.info(f'Average accurage of top L long range contacts is {average_acc}')
        for d in datasets:
            accuracy = ds_accuracy(dataset=d, model=get_model_by_name(model))
            LOGGER.info(f'Accuracy for {d}:\n{accuracy}')
            accuracy.to_csv(os.path.join(PATHS.models, model, 'predictions', f'{d}.csv'))
            _plot_acc_vs_msa(d, model)

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
