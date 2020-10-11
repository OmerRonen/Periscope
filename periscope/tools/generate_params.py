import os
from argparse import ArgumentParser

from ..net.params import NetParams
from ..utils.constants import PATHS, ARCHS, FEATURES


def parse_args():
    parser = ArgumentParser(description="Generates params file")
    parser.add_argument('-o', '--override', action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    override = args.override
    params_file = os.path.join(PATHS.periscope, 'params.yml')
    if override:
        os.remove(os.path.join(PATHS.periscope ,params_file))

    prms = NetParams(params_file,
                     conv_features=[],
                     name='model',
                     arch=ARCHS.multi_structure_ccmpred,
                     k=10,
                     batch_size=1,
                     num_layers=30,
                     filter_shape=(7, 7),
                     lr=0.0001)

    prms.save()


if __name__ == '__main__':
    main()
