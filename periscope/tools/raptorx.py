import os
import subprocess

from argparse import ArgumentParser
from collections import namedtuple

import yaml

RAPTOR_PATH = '/vol/sci/bio/data/or.zuk/projects/ContactMaps/src/RaptorX-3DModeling/DL4DistancePrediction4/Scripts'
_periscope_path = '/vol/sci/bio/data/or.zuk/projects/ContactMaps/src/Periscope'
DATA_PATH = os.path.join(_periscope_path, 'data')
os.path.join(_periscope_path, 'data', 'proteins')
_proteins_path = os.path.join(_periscope_path, 'data', 'proteins')


def yaml_load(filename):
    with open(filename, 'r') as stream:
        data_loaded = yaml.load(stream)

    return data_loaded


Datasets = namedtuple('Datasets', 'train eval pfam testing cameo membrane cameo41')

datasets = Datasets(
    train=
    yaml_load(os.path.join(DATA_PATH, 'valid',
                           'train.yaml'))['proteins'],
    eval=
    yaml_load(os.path.join(DATA_PATH, 'valid',
                           'eval.yaml'))['proteins'],

    pfam=
    yaml_load(os.path.join(DATA_PATH, 'valid',
                           'pfam.yaml'))['proteins'],

    cameo=yaml_load(os.path.join(DATA_PATH, 'valid',
                                 'cameo.yaml'))['proteins'],

    cameo41=yaml_load(os.path.join(DATA_PATH, 'valid',
                                   'cameo41.yaml'))['proteins'],

    membrane=yaml_load(os.path.join(DATA_PATH, 'valid',
                                    'membrane.yaml'))['proteins'],

    testing=['1ej0A', '1hh8A', '1kw4A', '1mk0A', '1tqgA', '1fl0A', '1jo8A']

)


def get_target_path(target):
    f_name = target
    t_path = os.path.join(_proteins_path, target[1:3], f_name)
    return t_path


def get_a3m_fname(target):
    target_hhblits_path = os.path.join(get_target_path(target), 'hhblits')
    a3m_file = os.path.join(target_hhblits_path, '%s.a3m' % target)
    return a3m_file


def parse_args():
    parser = ArgumentParser(description="Predict using RaptorX")
    parser.add_argument('-d', '--dataset', type=str, help='dataset name', default=None)
    parser.add_argument('-p', '--proteins', nargs="+", help='proteins', default=[])

    return parser.parse_args()


def _run_raptorx(target):
    a3m_file = get_a3m_fname(target)
    if not os.path.isfile(a3m_file):
        return None
    script = os.path.join(RAPTOR_PATH, 'PredictPairRelationFromMSA.sh')
    result_dir = os.path.join(DATA_PATH, 'raptorx')
    cmd = "%s -d %s %s" % (script, result_dir, a3m_file)
    subprocess.call(cmd, shell=True)


def main():
    args = parse_args()
    dataset = args.dataset
    print(dataset)
    if dataset is None:
        targets = args.proteins
    else:
        targets = getattr(datasets, dataset)
    for t in targets:
        _run_raptorx(t)


if __name__ == '__main__':
    main()
