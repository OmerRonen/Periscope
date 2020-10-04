import os
from argparse import ArgumentParser

from .protein_net import NetParams
from .globals import FEATURES, ARCHS


def parse_args():
    parser = ArgumentParser(description="Generates params file")
    parser.add_argument('-o', '--override', action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    override = args.override
    if override:
        os.remove('slurm_scripts/params.yml')

    prms = NetParams('slurm_scripts/params.yml',
                     conv_features=[],
                     name='template',
                     arch=ARCHS.multi_structure,
                     resnet_features=[],
                     k=7,
                     batch_size=1,
                     num_layers=12,
                     filter_shape=(7, 7),
                     lr=0.00005)

    prms.save()


if __name__ == '__main__':
    main()
