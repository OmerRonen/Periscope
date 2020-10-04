from argparse import ArgumentParser

from ..analysis.analyzer import plot_models_accuracy_per_dist


def parse_args():
    parser = ArgumentParser(description="creates structures dist plot")
    parser.add_argument('-m', '--model', type=str, help='main model')
    parser.add_argument('-r', '--ref', type=str, help='ref model')
    parser.add_argument('-d', '--dataset', type=str, help='dataset')

    return parser.parse_args()


def main():
    args = parse_args()
    main_model = args.model
    ref_model = args.ref
    dataset = args.dataset

    plot_models_accuracy_per_dist(model1=main_model, model2=ref_model, dataset=dataset)


if __name__ == '__main__':
    main()
