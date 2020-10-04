import logging
import os

import numpy as np

from ..data.creator import DataCreator
from ..utils.constants import PATHS, DATASETS_FULL, ERRS
from ..utils.utils import yaml_save, pkl_save, check_path

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def _get_valid_proteins(dataset, aln_method='old'):
    full_dataset = set(getattr(DATASETS_FULL, dataset))
    if aln_method == 'new':
        valid_proteins = [p for p in full_dataset if DataCreator(p).has_refs_new]
    else:
        valid_proteins = [p for p in full_dataset if DataCreator(p).has_refs]
    return valid_proteins


def _save_valid_datasets():
    valid_path = os.path.join(PATHS.periscope, 'data', 'valid')
    check_path(valid_path)

    pfam_valid = _get_valid_proteins("pfam")
    cameo_valid = _get_valid_proteins("cameo")
    cameo41_valid = _get_valid_proteins("cameo41")
    membrane_valid = _get_valid_proteins("membrane")

    yaml_save(os.path.join(valid_path, 'pfam.yaml'), data={'proteins': pfam_valid})
    yaml_save(os.path.join(valid_path, 'cameo.yaml'), data={'proteins': cameo_valid})
    yaml_save(os.path.join(valid_path, 'cameo41.yaml'), data={'proteins': cameo41_valid})
    yaml_save(os.path.join(valid_path, 'membrane.yaml'), data={'proteins': membrane_valid})


def _save_train_eval_valid(train, test):
    train -= test
    valid = np.array(_get_valid_proteins(train))

    np.random.shuffle(valid)
    percentile = int(np.ceil(0.8 * len(valid)))
    train_valid, eval_valid = list(valid[:percentile]), list(valid[percentile:])

    pkl_save(os.path.join(PATHS.periscope, 'data', 'train_valid_2.pkl'), data={'proteins': train_valid})
    yaml_save(os.path.join(PATHS.periscope, 'data', 'train_valid_2.yaml'), data={'proteins': train_valid})
    pkl_save(os.path.join(PATHS.periscope, 'data', 'eval_valid_2.pkl'), data={'proteins': eval_valid})
    yaml_save(os.path.join(PATHS.periscope, 'data', 'eval_valid_2.yaml'), data={'proteins': eval_valid})


def _get_data(dataset):
    return set(getattr(DATASETS_FULL, dataset))


def main():
    aln_method = 'old'
    pfam = _get_data('pfam')
    cameo = _get_data('cameo')
    cameo41 = _get_data('cameo41')
    membrane = _get_data('membrane')

    test = cameo | pfam | membrane | cameo41
    train = _get_data('train')
    train |= _get_data('eval')
    train -= test
    _save_valid_datasets()
    valid = []
    errs = []
    if aln_method == 'new':
        for p in train:
            try:
                if DataCreator(p).has_refs_new:
                    valid.append(p)
            except Exception:
                errs.append(p)
                pass
        LOGGER.info(f'erros: {errs}')
    else:
        for p in train:
            try:
                if DataCreator(p).has_refs:
                    valid.append(p)
            except Exception:
                errs.append(p)
                pass
        LOGGER.info(f'erros: {errs}')

    np.random.shuffle(valid)
    percentile = int(np.ceil(0.8 * len(valid)))
    train_valid, eval_valid = list(valid[:percentile]), list(valid[percentile:])
    valid_path = os.path.join(PATHS.periscope, 'data', 'valid')

    yaml_save(os.path.join(valid_path, 'train.yaml'), data={'proteins': train_valid})
    yaml_save(os.path.join(valid_path, 'eval.yaml'), data={'proteins': eval_valid})


if __name__ == '__main__':
    main()
