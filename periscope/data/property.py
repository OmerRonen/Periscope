import os
import subprocess
import tempfile
from collections import namedtuple

import numpy as np

from periscope.utils.constants import PATHS
from periscope.utils.utils import (get_target_hhblits_path, get_target_path, pkl_load, get_a3m_fname, get_aln_fasta,
                                   get_target_dataset, check_path, pkl_save, get_fasta_fname, write_fasta)

CATEGORIES = {'ss8': ["H", "G", "I", "E", "B", "T", "S", "L"],
              "diso": [".", "*"],
              "acc": ['B', 'M', 'E']}

Features = namedtuple(
    'Propeties',
    'ss8 diso acc')


def _get_property_path(target, family=None):
    return os.path.join(get_target_path(target, family=family), "property")


def run_tgt(target, family, tgt_file, msa):
    # dc = DataCreator(target, family=family, train=False)

    fasta_file = get_fasta_fname(target, family)
    a3m_aln = get_a3m_fname(target, family=family)

    if not os.path.isfile(a3m_aln):
        reformat = f'reformat.pl {get_aln_fasta(target, family=family)} {a3m_aln}'
        subprocess.call(reformat, shell=True)
    d = tempfile.TemporaryDirectory()
    if family:
        tmp_fasta = os.path.join(d.name, 'f.fasta')
        subprocess.run(f"cp {fasta_file} {tmp_fasta}", shell=True)
        fasta_file = tmp_fasta
        temp_fasta = os.path.join(d.name, 'msa.fasta')
        write_fasta(msa, temp_fasta)
        tmp_a3m = os.path.join(d.name, 'f.a3m')

        reformat = f'reformat.pl {temp_fasta} {tmp_a3m}'
        subprocess.call(reformat, shell=True)
        # subprocess.run(f"cp {a3m_aln} {tmp_a3m}",shell=True)
        a3m_aln = tmp_a3m

    cmd = f'A3M_To_TGT -i {fasta_file} -I {a3m_aln} -o {tgt_file} -t {d.name}'
    subprocess.run(cmd, shell=True, cwd=os.path.join(PATHS.src, 'TGT_Package'))


def _run_property(target, family=None, msa=None):
    tgt_path = os.path.join(PATHS.periscope, 'data', get_target_dataset(target, family=family))
    check_path(tgt_path)
    tgt_file = os.path.join(tgt_path, f'{target}.tgt')
    run_tgt(target,family, tgt_file, msa=msa)
    predict_property = os.path.join(PATHS.src, "Predict_Property", "Predict_Property.sh")
    property_path = _get_property_path(target, family)
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


def _get_prop_fasta(prop, target, family=None):
    return os.path.join(_get_property_path(target, family), f'{target}_{prop}')


def get_properties(target, train, family=None, msa=None):
    pth = _get_property_path(target, family)
    if not os.path.exists(pth):
        if train:
            return
        _run_property(target, family, msa=msa)
    prop_arr = []
    for p in ['acc', 'ss8', 'diso']:
        fl = os.path.join(pth, f"{target}.{p}_simp")
        if not os.path.isfile(fl):
            return
        prop_arr.append(load_fasta(fl))
    try:
        return np.concatenate(prop_arr, axis=1)
    except Exception:
        return


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
