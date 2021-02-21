import itertools
import logging
import os
import subprocess
import tempfile
from shutil import copyfile

import numpy as np
from Bio import pairwise2

from .utils import (check_path, get_fasta_fname, get_target_dataset, yaml_save, get_modeller_pdb_file, get_aln_fasta,
                    yaml_load, get_a3m_fname, save_chain_pdb, pkl_save, get_target_path, pkl_load,
                    get_target_hhblits_path)
from ..analysis.analyzer import get_model_predictions
from ..data.creator import DataCreator
from ..data.property import get_raptor_ss3_txt
from ..net.contact_map import ContactMapEstimator, get_model_by_name
from ..utils.constants import PATHS, DATASETS
from ..utils.protein import Protein

NOE_HEADER = 'TOTAL  VDW    BOND   NOE    2.5 3.5  H   E    CONTACTS  SS-NOE    HBONDS    DIHEDRAL   CONTACTS SS-NOE   HBONDS\n'

logging.basicConfig(level=logging.DEBUG)

LOGGER = logging.getLogger(__name__)


def model_modeller_tm_scores(model_name, target, fast=False, sswt=5, selectrr='2.0L'):
    dataset = get_target_dataset(target)
    dc = DataCreator(target)
    if not dc.has_refs:
        return

    model = get_model_by_name(model_name, dataset)
    logits = model.predict(proteins=[target], dataset=dataset)['logits'].get(target, None)

    # logits = get_model_predictions(model, proteins=[target])
    if logits is None:
        return
    dataset = model.predict_data_manager.dataset if dataset is None else dataset
    _save_cns_data(model, {target: logits}, dataset)
    target_tm_f = _get_target_tm_fast if fast else _get_target_tm
    LOGGER.info(f'{model_name} score for {target}:\n')
    tm_model = target_tm_f(target, model, sswt=sswt, selectrr=selectrr)

    LOGGER.info(f'modeller score for {target}:\n')
    templates = True
    modeller_pdb = get_modeller_pdb_file(target, templates=templates, n_struc=1, sp=False)
    if not os.path.isfile(modeller_pdb):
        dc.run_modeller_templates(n_structures=1)

    tm_modeller = get_modeller_tm_score(target, templates=templates)
    LOGGER.info(f'modeller score with starting point for {target}:\n')
    tm_modeller_sp = get_modeller_tm_score(target, templates=True, sp=True)
    LOGGER.info(f"Ref is {dc.closest_pdb}")
    LOGGER.info(f'reference score for {target}:\n')
    tm_ref = get_ref_tm_score(target, dc.closest_pdb)

    scores_file = os.path.join(model.path, f'tm_{sswt}_{selectrr}', dataset, f'{target}.yaml')
    check_path(os.path.join(model.path, f'tm_{sswt}_{selectrr}', dataset))
    dc = DataCreator(target)
    n_refs = dc.n_refs_test
    n_homs = dc.n_homs

    scores = {model.name: tm_model, 'modeller': tm_modeller, 'modeller_sp': tm_modeller_sp,
              'n_refs': n_refs, "n_homs": n_homs, 'ref': tm_ref}
    yaml_save(data=scores, filename=scores_file)


def save_modeller_scores(dataset):
    targets = getattr(DATASETS, dataset)
    scores = {}
    for target in targets:
        LOGGER.info(target)
        dc = DataCreator(target, 10)
        if dc.sorted_structures is None:
            continue
        try:
            modeller_pdb = get_modeller_pdb_file(target, templates=True, n_struc=1)
            if not os.path.isfile(modeller_pdb):
                dc.run_modeller_templates(1)
            tm_old = get_modeller_tm_score(target=target)
            tm_new = get_modeller_tm_score(target=target, templates=True)
        except Exception:
            continue

        scores[target] = {'old': tm_old, 'new': tm_new}
        dataset_path = os.path.join(PATHS.models, 'modeller', dataset)
        yaml_save(data=scores, filename=os.path.join(dataset_path, f'tm_scores.yaml'))


def _get_tm_score(native_pdb, predicted_pdb):
    with tempfile.NamedTemporaryFile() as f:
        outfile = f.name
        cmd_log = f"{os.path.join(PATHS.src, 'TMscore')} {native_pdb} {predicted_pdb}"
        cmd = f"{os.path.join(PATHS.src, 'TMscore')} {native_pdb} {predicted_pdb}  > {outfile}"
        subprocess.run(cmd_log, shell=True)
        subprocess.run(cmd, shell=True)
        tm_score = _parse_tm_score(outfile)
    return tm_score


def _print_tm_score(native_pdb, predicted_pdb):
    cmd = f"{os.path.join(PATHS.src, 'TMscore')} {native_pdb} {predicted_pdb}"
    subprocess.run(cmd, shell=True)


def get_ref_tm_score(target, reference):
    pdb = Protein(target[0:4], target[4]).pdb_fname
    ref_pdb = Protein(reference[0:4], reference[4]).pdb_fname

    pred_pdb = f'{target}_pred.pdb'
    save_chain_pdb(reference, pred_pdb, ref_pdb, 0)

    native_pdb = f'{target}_native.pdb'
    save_chain_pdb(target, native_pdb, pdb, 0)

    tm = _get_tm_score(native_pdb=native_pdb, predicted_pdb=pred_pdb)
    os.remove(pred_pdb)
    os.remove(native_pdb)
    return tm


def get_modeller_tm_score(target, templates=False, sp=False):
    protein, chain = target[0:4], target[4]
    modeller_pdb = get_modeller_pdb_file(target, templates=templates, n_struc=1, sp=sp)
    if not os.path.isfile(modeller_pdb):
        return

    pdb = Protein(protein, chain).pdb_fname

    pred_pdb = f'{target}_pred.pdb'
    save_chain_pdb(target, pred_pdb, modeller_pdb, 0, skip_chain=templates)

    native_pdb = f'{target}_native.pdb'
    save_chain_pdb(target, native_pdb, pdb, 0, old=not templates)

    tm = _get_tm_score(native_pdb=native_pdb, predicted_pdb=pred_pdb)
    os.remove(pred_pdb)
    os.remove(native_pdb)
    return tm


def _parse_tm_score(log_file):
    with open(log_file, 'r') as f:
        lines = list(f.readlines())

    tm_score = [line for line in lines if line.startswith('TM-score')][0].split(' ')[5]
    return float(tm_score)


def _save_rr_file(model: ContactMapEstimator, full):
    dataset = model.predict_data_manager.dataset
    predictions = get_model_predictions(model, dataset)
    model_rr_path = os.path.join(PATHS.periscope, 'data', dataset, model.path.split('/')[-1])
    check_path(model_rr_path)
    for target in predictions:
        predicted_logits = predictions[target]['logits']
        txt = _get_rr_txt(predicted_cm=predicted_logits, target=target, full=full)
        with open(os.path.join(model_rr_path, target + '_full.rr'), 'w') as f:
            f.write(txt)





def dist(i, j):
    return np.abs(i - j)


def _get_mask(l):
    mask = np.fromfunction(dist, shape=(l, l))
    mask = np.where(mask > 5, 1, 0)
    return mask


def _get_n_contacts_from_refs(target):
    p = Protein(target[0:4], target[4])
    dc = DataCreator(target, 10)
    mask = _get_mask(p.dm.shape[0])
    n_contacts = np.round((np.sum(dc.reference_dm * mask) / 2) / p.dm.shape[0], 1)
    return n_contacts


def _get_rr_txt(predicted_cm, target, full=False):
    p = Protein(target[0:4], target[4])
    if full:
        alignment = pairwise2.align.globalms(p.str_seq, p.str_seq_full, 5, -.4, -.5, -.1, one_alignment_only=True)[0]
        long_inds = [i for i in range(len(p.str_seq_full)) if alignment[0][i] != '-']
        short_inds = list(range(len(p.str_seq)))

        pairs_long = list(itertools.combinations(long_inds, 2))
        pairs_short = list(itertools.combinations(short_inds, 2))

        pairs_long, pairs_short = list(zip(*pairs_long)), list(zip(*pairs_short))
        l = len(p.str_seq_full)
        extended_prediction = np.zeros((l, l))
        extended_prediction[pairs_long[0], pairs_long[1]] = predicted_cm[pairs_short[0], pairs_short[1]]

        content = p.str_seq_full + "\n"
        prediction = np.round(extended_prediction, 6)
    else:
        content = p.str_seq + "\n"
        l = len(p.str_seq)
        prediction = np.round(predicted_cm, 6)
    table = {}
    all_logits = []
    for i in range(0, l - 6):
        for j in range(i + 6, l):
            all_logits.append(prediction[i, j])

    for i in range(0, l - 6):
        for j in range(i + 6, l):

            lgt = prediction[i, j]

            logit = '{:f}'.format(lgt)

            if prediction[i, j] < 0.2:
                continue
            if logit in table:
                table[logit] += f'{i + 1} {j + 1} 0 8 {logit}\n'
            else:
                table[logit] = f'{i + 1} {j + 1} 0 8 {logit}\n'
    sorted_content = "".join([table[k] for k in sorted(table.keys(), reverse=True)])
    content += sorted_content
    return content


def _get_model_cns_path(model, dataset):
    return os.path.join(PATHS.periscope, 'data', 'cns_refs_contacts', dataset, model.name)


def _save_cns_data(model: ContactMapEstimator, predictions, dataset):
    model_cns_path = _get_model_cns_path(model, dataset)
    check_path(model_cns_path)
    for target, logits in predictions.items():
        suffix_rr = '.rr'
        rr_file = os.path.join(model_cns_path, target + suffix_rr)
        if not os.path.isfile(rr_file):
            txt_rr = _get_rr_txt(predicted_cm=np.squeeze(logits), target=target)

            with open(rr_file, 'w') as f:
                f.write(txt_rr)
        suffix_ss = '.ss'

        ss_file = os.path.join(model_cns_path, target + suffix_ss)
        if not os.path.isfile(ss_file):
            txt_ss = get_raptor_ss3_txt(target)

            with open(ss_file, 'w') as f:
                f.write(txt_ss)


def _run_cns(target, model, outdir, sswt, selectrr, dataset, full):
    model_cns_path = _get_model_cns_path(model, dataset)
    confold = os.path.join(PATHS.src, 'confold_v1.0', 'confold.pl')
    suffix_ss = '.ss'
    suffix_rr = '.rr'

    rr_file = os.path.join(model_cns_path, target + suffix_rr)
    ss_file = os.path.join(model_cns_path, target + suffix_ss)
    fasta_file = get_fasta_fname(target, full)
    f_pdb = os.path.join(outdir, 'stage1', f'{target}_model1.pdb')

    selectrr = f'{_get_n_contacts_from_refs(target)}L' if selectrr.isdigit() else selectrr

    if os.path.isfile(f_pdb):
        return
    cmd = f'{confold} -rr {rr_file} -ss {ss_file} -seq {fasta_file} -sswt {sswt}  -mcount 20 -o {outdir} --selectrr {selectrr}'
    LOGGER.info(cmd)
    subprocess.run(cmd, shell=True)


def _get_target_tm(target, model, full=False, sswt=5, dataset=None, selectrr='2.0L'):
    dataset = model.predict_data_manager.dataset if dataset is None else dataset
    outdir = os.path.join(model.path, 'cns', dataset, target, 'stage1')
    predicted_pdb = os.path.join(outdir, f'{target}_model1.pdb')

    check_path(outdir)
    with tempfile.TemporaryDirectory() as d:
        _run_cns(target, model, d, sswt, selectrr, dataset, full)
        predicted_pdb_tmp = os.path.join(d, 'stage1', f'{target}_model1.pdb')
        copyfile(predicted_pdb_tmp, predicted_pdb)

    pred_pdb = f'{target}_pred.pdb'
    save_chain_pdb(target, pred_pdb, predicted_pdb, 0)

    native_pdb = f'{target}_native.pdb'
    save_chain_pdb(target, native_pdb, Protein(target[0:4], target[4]).pdb_fname, 0)

    tm = _get_tm_score(native_pdb, pred_pdb)
    os.remove(pred_pdb)
    os.remove(native_pdb)
    return tm


def _get_target_tm_fast(target, model, sswt=5, dataset=None, selectrr='2.0L'):
    dataset = model.predict_data_manager.dataset if dataset is None else dataset

    outdir = os.path.join(model.path, f'cns_{sswt}_{selectrr.replace(".", "_")}', dataset, target)

    if not os.path.exists(outdir):
        return
    try:
        predicted_pdb = os.path.join(outdir, 'stage1', f'{target}_model1.pdb')
        pred_pdb = f'{target}_pred.pdb'
        save_chain_pdb(target, pred_pdb, predicted_pdb, 0)

        native_pdb = f'{target}_native.pdb'
        save_chain_pdb(target, native_pdb, Protein(target[0:4], target[4]).pdb_fname, 1)

        tm = _get_tm_score(native_pdb, predicted_pdb)
        os.remove(pred_pdb)
        os.remove(native_pdb)
    except FileNotFoundError:
        return
    return tm
