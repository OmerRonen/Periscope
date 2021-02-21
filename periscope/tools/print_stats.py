import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from argparse import ArgumentParser
from numpy.polynomial.polynomial import polyfit

from matplotlib.lines import Line2D
from scipy import interpolate

from ..analysis.artist import get_cm
from ..utils.utils import get_data, get_raptor_logits
from ..utils.constants import PATHS, yaml_load, DATASETS
from ..analysis.analyzer import ds_accuracy, calc_accuracy, accuracy_short
from ..net.contact_map import get_model_by_name
from ..analysis.stats import get_datasets_pre_post

LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser(description="generates data for taget")
    parser.add_argument('model')
    parser.add_argument('-d', '--datasets', nargs='+', default=None)

    return parser.parse_args()


def _get_templates_relative_contribution(data):
    prediction = data['prediction']
    l = int(prediction.shape[0])
    cm, _ = get_cm(prediction, l)
    w_templates = np.sum(data['weights'][..., 0:10], axis=-1)
    w_evo = data['weights'][..., 10]
    ratio = w_templates / (w_evo + w_templates)
    return np.mean(ratio[cm == 1])


def _plot_acc_vs_msa(dataset, model_name):
    targets = getattr(DATASETS, dataset)
    plot_data = {"Periscope": [], "RaptorX": [], "beff": [], "# of Templates": [], "Templates Relative Weight": []}
    for t in targets:
        data = get_data(model_name, t)
        if data is None:
            continue
        raptor_logits = get_raptor_logits(t)
        logits = data['prediction']
        gt = data['gt']
        acc = calc_accuracy(pred=logits, gt=gt, top=0.5)
        acc_raptor = calc_accuracy(pred=raptor_logits, gt=gt, top=0.5) if raptor_logits is not None else 0

        plot_data['Periscope'].append(acc)
        plot_data['RaptorX'].append(acc_raptor)
        plot_data['beff'].append(np.log(float(data['beff'])))
        templates = data['templates']
        n_refs = 0
        if templates is not None:
            n_refs = np.sum(np.max(np.max(templates, axis=0), axis=0) > 0)
        plot_data['# of Templates'].append(n_refs)
        plot_data['Templates Relative Weight'].append(_get_templates_relative_contribution(data))

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
    plt.figure(figsize=(7, 7))
    fig = plt.gcf()
    fig.suptitle(f"{dataset} predictions", fontsize=14)
    plt.subplot(321)
    plot_df_smooth = _get_smooth_df('beff', plot_df)
    plt.scatter('beff', 'Periscope', data=plot_df_smooth, marker='o', alpha=0.4, color='blue')
    b, m = polyfit(plot_df_smooth.beff, plot_df_smooth.Periscope, 1)
    plt.plot(plot_df_smooth.beff, b + m * plot_df_smooth.beff, '--', color='blue', alpha=0.6)
    plt.scatter('beff', 'RaptorX', data=plot_df_smooth, marker='o', alpha=0.4, color="orange")
    b, m = polyfit(plot_df_smooth.beff, plot_df_smooth.RaptorX, 1)
    plt.plot(plot_df_smooth.beff, b + m * plot_df_smooth.beff, '--', color='orange', alpha=0.6)
    plt.xlabel('$\log(Beff)$')
    plt.ylabel('Accuracy')
    custom_lines = [Line2D([0], [0], color='blue', lw=4),
                    Line2D([0], [0], color='orange', lw=4)]
    plt.legend(custom_lines, ['Periscope', 'RaptorX'], loc=2, prop={'size': 8})
    plot_df_smooth = _get_smooth_df('# of Templates', plot_df)

    plt.subplot(322)
    n_temps = np.unique(plot_df_smooth['# of Templates'])
    data_raptor = [list(plot_df_smooth.RaptorX[plot_df_smooth['# of Templates'] == n]) for n in n_temps]
    data_periscope = [list(plot_df_smooth.Periscope[plot_df_smooth['# of Templates'] == n]) for n in n_temps]

    ticks = n_temps

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    bpl = plt.boxplot(data_periscope, positions=np.array(range(len(data_periscope))) * 2.0 - 0.4, sym='', widths=0.6)
    bpr = plt.boxplot(data_raptor, positions=np.array(range(len(data_raptor))) * 2.0 + 0.4, sym='', widths=0.6)
    set_box_color(bpl, 'blue')  # colors are from http://colorbrewer2.org/
    set_box_color(bpr, 'orange')

    # draw temporary red and blue lines and use them to create a legend
    plt.xlabel('# of Templates')
    # plt.ylabel('Accuracy')
    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks) * 2)
    plt.ylim(0, 1)
    plt.tight_layout()
    # plt.scatter('# of Templates', 'Periscope', data=plot_df_smooth, marker='o', alpha=0.4, color='blue')
    # plt.scatter('# of Templates', 'RaptorX', data=plot_df_smooth, marker='o', alpha=0.4, color="orange")
    # plt.ylabel('Accuracy')
    # plt.xlabel('# of Templates')

    plt.subplot(323)
    plot_df_smooth = _get_smooth_df('beff', plot_df)

    plt.scatter('beff', 'Templates Relative Weight', data=plot_df_smooth, marker='o', alpha=0.4, color='blue')
    plt.xlabel('$\log(Beff)$')
    plt.ylabel('Templates Relative Weight')
    plt.savefig(os.path.join(PATHS.models, model_name, 'predictions', f'{dataset}_msa_accuracy.png'))


def main():
    # print(get_datasets_pre_post())
    args = parse_args()
    model = args.model
    datasets = args.datasets
    accuracy_short(get_model_by_name(model), datasets)
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
