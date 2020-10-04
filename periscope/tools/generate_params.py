import os
from argparse import ArgumentParser

from ..net.params import NetParams
from ..utils.constants import PATHS, ARCHS


def parse_args():
    parser = ArgumentParser(description="Generates params file")
    parser.add_argument('-o', '--override', action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    override = args.override
    if override:
        os.remove(os.path.join(PATHS.periscope ,'slurm_scripts/params.yml'))

    prms = NetParams('slurm_scripts/params.yml',
                     conv_features=[],
                     name='template',
                     arch=ARCHS.conv,
                     resnet_features=[],
                     k=10,
                     batch_size=1,
                     num_layers=30,

                     filter_shape=(7, 7),
                     lr=0.0001)

    prms.save()


if __name__ == '__main__':
    main()
