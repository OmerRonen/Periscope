import os
import random
import shutil
import logging

from .creator import DataCreator
from .seeker import DataSeeker
from ..utils.constants import PATHS, DATASETS
from ..utils.utils import check_path, get_pdb_fname, get_target_path, get_target_ccmpred_file, get_target_evfold_file, \
    get_aln_fasta, get_clustalo_aln

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def clean_pdb(target):
    src = '/cs/zbio/orzuk/projects/ContactMaps/data/pdb'

    def _mv_protein(pdb_file):
        protein = pdb_file.split('pdb')[1].split('.ent')[0]
        new_path = os.path.join(PATHS.data, 'pdb', protein[1:3])
        check_path(new_path)
        src_file = os.path.join(src, pdb_file)
        dst_file = get_pdb_fname(protein)
        if os.path.isfile(src_file) and not os.path.isfile(dst_file):
            LOGGER.info(f'Moving {src_file} to {dst_file}')
            shutil.move(src_file, dst_file)

    pdb_files_full = []
    for folder in os.listdir(src):
        pdb_files_full += os.listdir(os.path.join(src, folder))

    pdb_files = [f'pdb{target[0:4]}.ent'] if target is not None else pdb_files_full

    for pdb_file in pdb_files:
        _mv_protein(pdb_file)


def _remove_file(fname):
    if os.path.isfile(fname):
        LOGGER.info(f'Removing {fname}')
        os.remove(fname)


def clean_hhblits():
    train = set(DATASETS.train) | set(DATASETS.eval)
    test = set(DATASETS.pfam) | set(DATASETS.membrane) | set(DATASETS.cameo) | set(DATASETS.cameo41)
    all_targets = train | test
    for t in all_targets:
        t_path = os.path.join(get_target_path(t), 'hhblits')
        a3m_file = os.path.join(t_path, f'{t}.a3m')
        a2m_file = os.path.join(t_path, f'{t}.a2m')

        _remove_file(a3m_file)
        _remove_file(a2m_file)
    for t in train:
        fasta_aln = get_aln_fasta(t)
        clustalo_aln = get_clustalo_aln(t)
        evfold_path = os.path.join(get_target_path(t), 'evfold')
        evfold_params = os.path.join(evfold_path, f'{t}_2.params')
        if not os.path.exists(evfold_path):
            continue
        if len(os.listdir(evfold_path)) > 1 and os.path.isfile(clustalo_aln):
            _remove_file(fasta_aln)
            _remove_file(evfold_params)


def clean_structures(target):
    targets = os.listdir(PATHS.msa_structures) if target is None else [target]
    for target in targets:
        if target not in os.listdir(PATHS.msa_structures):
            continue
        protein_path = os.path.join(PATHS.proteins, target[1:3], target, 'features')
        check_path(protein_path)
        struc_path = os.path.join(PATHS.msa_structures, target)
        if not os.path.exists(struc_path):
            continue
        old_features = os.listdir(struc_path)
        for file in old_features:
            shutil.move(os.path.join(PATHS.msa_structures, target, file), os.path.join(protein_path, file))
        shutil.rmtree(os.path.join(PATHS.msa_structures, target))


def clean_ccmpred(target):
    files = os.listdir(PATHS.ccmpred) if target is None else [f'{target}.mat']
    for file in files:
        target = file.split('.')[0]
        ccmpred_target_path = os.path.join(get_target_path(target), 'ccmpred')
        check_path(ccmpred_target_path)
        src_file = os.path.join(PATHS.ccmpred, file)
        if not os.path.isfile(src_file):
            continue
        dst_file = os.path.join(ccmpred_target_path, file)
        LOGGER.info(f'Moving {src_file} to {dst_file}')
        shutil.move(src_file, dst_file)


def clean_evfold(target):
    targets = {f.split('_')[0] for f in os.listdir(PATHS.evfold) if f.endswith('v2.pkl')} if target is None else [
        target]
    for target in targets:
        evfold_target_path = os.path.join(get_target_path(target), 'evfold')
        check_path(evfold_target_path)

        parmas_file = f'{target}_2.params'
        src_params = os.path.join(PATHS.evfold, parmas_file)
        dst_params = os.path.join(evfold_target_path, parmas_file)
        if os.path.isfile(src_params):
            LOGGER.info(f'Moving {src_params} to {dst_params}')

            shutil.move(src_params, dst_params)

        txt_file = f'{target}_v2.txt'
        src_txt = os.path.join(PATHS.evfold, txt_file)
        dst_txt = os.path.join(evfold_target_path, txt_file)
        if os.path.isfile(src_txt):
            LOGGER.info(f'Moving {src_txt} to {dst_txt}')

            shutil.move(src_txt, dst_txt)

        pkl_file = f'{target}_v2.pkl'
        src_pkl = os.path.join(PATHS.evfold, pkl_file)
        dst_pkl = os.path.join(evfold_target_path, pkl_file)
        if os.path.isfile(src_pkl):
            LOGGER.info(f'Moving {src_pkl} to {dst_pkl}')

            shutil.move(src_pkl, dst_pkl)


def clean_sifts():
    shutil.move('/cs/zbio/orzuk/projects/ContactMaps/data/MSA_Completion/sifts_mapping.pkl',
                os.path.join(PATHS.data, 'sifts_mapping.pkl'))


def clean_query(target):
    fasta_files = [f for f in os.listdir(os.path.join(PATHS.msa, 'query')) if f.endswith('.fasta')]
    fasta_files = fasta_files if target is None else [f'{target}.fasta']
    random.shuffle(fasta_files)
    for file in fasta_files:
        src_fasta = os.path.join(os.path.join(PATHS.msa, 'query', file))
        target_fasta = os.path.join(get_target_path(file.split('.')[0]), file)
        check_path(get_target_path(file.split('.')[0]))
        LOGGER.info(f'Moving {src_fasta} to {target_fasta}')
        try:
            shutil.move(src_fasta, target_fasta)
        except FileNotFoundError:
            pass

    hmm_files = [f for f in os.listdir(os.path.join(PATHS.msa, 'query')) if f.endswith('.hhr')]
    hmm_files = hmm_files if target is None else [f'{target}.hhr']
    random.shuffle(hmm_files)

    for file in hmm_files:
        hh_path = os.path.join(get_target_path(file.split('.')[0]), 'hhblits')
        check_path(hh_path)
        src_hmm = os.path.join(os.path.join(PATHS.msa, 'query', file))
        target_hmm = os.path.join(hh_path, file)
        LOGGER.info(f'Moving {src_hmm} to {target_hmm}')
        try:
            shutil.move(src_hmm, target_hmm)
        except FileNotFoundError:
            pass


def clean(target=None):
    # clean_pdb(target)
    # clean_ccmpred(target)
    # clean_evfold(target)
    # clean_hhblits(target)
    # clean_sifts()
    # clean_structures(target)
    clean_hhblits()


def check_clean(target):
    had_ccmpred = os.path.isfile(os.path.join(PATHS.ccmpred, f'{target}.mat'))
    has_evfold = os.path.isfile(os.path.join(PATHS.evfold, f'{target}_v2.pkl'))
    struc_path = os.path.join(PATHS.msa_structures, target)
    old_features = os.listdir(struc_path) if os.path.exists(struc_path) else []
    clean(target)
    ds = DataSeeker(target, 10)
    dc = DataCreator(target, 10)

    assert os.path.isfile(ds.protein.pdb_fname)
    assert os.path.isfile(dc.protein.pdb_fname)

    if had_ccmpred:
        assert os.path.isfile(get_target_ccmpred_file(target))
        assert ds.ccmpred is not None
        assert dc.ccmpred is not None

    if has_evfold:
        assert os.path.isfile(get_target_evfold_file(target))
        assert ds.evfold is not None
        assert dc.evfold is not None

    assert os.path.isfile(os.path.join(get_target_path(target), f'{target}.fasta'))

    if 'structures_sorted.pkl' in old_features:
        assert dc.sorted_structures is not None
        assert ds.sorted_structures is not None
