import os
import subprocess
import tempfile
from collections import namedtuple

import numpy as np

from periscope.utils.constants import PATHS
from periscope.utils.utils import (get_target_hhblits_path, get_target_path, pkl_load, get_a3m_fname, get_aln_fasta,
                                   get_target_dataset, check_path, pkl_save)

CATEGORIES = {'ss8': ["H", "G", "I", "E", "B", "T", "S", "L"],
              "diso": [".", "*"],
              "acc": ['B', 'M', 'E']}

Features = namedtuple(
    'Propeties',
    'ss8 diso acc')


def _get_property_path(target):
    return os.path.join(get_target_path(target), "property")


def _run_property(target):
    pth = get_target_hhblits_path(target)
    fasta_file = os.path.join(pth, f'{target}.fasta')
    a3m_aln = get_a3m_fname(target)

    if not os.path.isfile(a3m_aln):
        reformat = f'reformat.pl {get_aln_fasta(target)} {a3m_aln}'
        subprocess.call(reformat, shell=True)
    tgt_path = os.path.join(PATHS.periscope, 'data', get_target_dataset(target))
    check_path(tgt_path)
    tgt_file = os.path.join(tgt_path, f'{target}.tgt')

    cmd = f'A3M_To_TGT -i {fasta_file} -I {a3m_aln} -o {tgt_file}'
    subprocess.run(cmd, shell=True, cwd=os.path.join(PATHS.src, 'TGT_Package'))
    os.remove(a3m_aln)

    predict_property = os.path.join(PATHS.src, "Predict_Property", "Predict_Property.sh")
    property_path = _get_property_path(target)
    cmd = f'{predict_property} -i {tgt_file} -o {property_path}'
    subprocess.run(cmd, shell=True)


def _catorical_arr(arr, cat):
    cat_arr = []
    for d in arr:
        cat_arr.append(np.array(d == np.array(cat), dtype=np.int))
    return np.stack(cat_arr)


def load_fasta(fname, return_txt=False):
    with open(fname, 'r') as f:
        txt = list(f.readlines())

    property = txt[2]

    property_txt = "".join([txt[0], property])
    if return_txt:
        return property_txt

    property_arr = np.array(list(property[:-1]))
    cat_name = fname.split(f'/')[-1].split('.')[-1].split('_')[0]
    categories = CATEGORIES[cat_name]
    return _catorical_arr(property_arr, categories)


def _get_prop_fasta(prop, target):
    return os.path.join(_get_property_path(target), f'{target}_{prop}')


def get_properties(target):
    fname = get_raptor_properties_fname(target)
    if os.path.isfile(fname):
        pkl_load(fname)
    pth = _get_property_path(target)
    if not os.path.exists(pth):
        _run_property(target)
    prop_arr = []
    for p in ['acc', 'ss8', 'diso']:
        fl = os.path.join(pth, f"{target}.{p}_simp")
        prop_arr.append(load_fasta(fl))
    prop_raptor = np.concatenate(prop_arr, axis=1)
    pkl_save(fname, prop_raptor)
    return prop_raptor


def get_raptor_ss_fname(target):
    return os.path.join(get_target_path(target), 'features', 'ss_raptor.pkl')


def get_raptor_properties_fname(target):
    return os.path.join(get_target_path(target), 'features', 'prop_raptor.pkl')


def get_raptor_acc_fname(target):
    return os.path.join(get_target_path(target), 'features', 'acc_raptor.pkl')


def get_raptor_ss(target):
    if not os.path.exists(get_target_hhblits_path(target)):
        return
    fname = get_raptor_ss_fname(target)
    if not os.path.isfile(fname):
        _ = get_raptor_ss3_txt(target)
    return pkl_load(fname)


def get_raptor_ss3_txt(target):
    property_path = _get_property_path(target)

    ss3_file = os.path.join(property_path, f'{target}.ss3_simp')

    with open(ss3_file, 'r') as f:
        ss3_txt = list(f.readlines())

    ss = ss3_txt[2]

    ss3_txt = "".join([ss3_txt[0], ss])
    pkl_save(data=np.array(list(ss[:-1])), filename=get_raptor_ss_fname(target))

    return ss3_txt
