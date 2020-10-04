import os
import pandas as pd

from argparse import ArgumentParser

from .analysis import plot_target_analysis


def _get_targets(model_name):
    targets = os.listdir('models/%s/artifacts/data' % model_name)
    return targets


def create_plots(model_name, target=None):
    targets = _get_targets(model_name) if not target else [target]

    figures_path = os.path.join('models', model_name, 'artifacts', 'figures')

    for t in targets:
        data_path = os.path.join('models', model_name, 'artifacts', 'data', t)
        logits = pd.read_csv(os.path.join(data_path, 'logits.csv'),
                             index_col=0).values
        plot_target_analysis(predicted_logits=logits,
                             data_path=data_path,
                             target=t,
                             figures_path=figures_path)


def parse_args():
    parser = ArgumentParser(
        description="Create plots for contact map predictior")
    parser.add_argument('-n', '--model-name', type=str, help='Model name')
    parser.add_argument('-t', '--target', type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model_name
    target = args.target

    create_plots(model_name, target)


if __name__ == '__main__':
    main()
