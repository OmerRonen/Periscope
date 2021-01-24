import logging
import os

import numpy as np

from argparse import ArgumentParser

from periscope.analysis.analyzer import get_top_category_accuracy_np
from periscope.data.creator import DataCreator
from periscope.net.contact_map import get_model_by_name, ContactMapEstimator
from periscope.utils.constants import DATASETS, DATASETS_FULL
from periscope.utils.drive import upload_folder
from periscope.utils.tm import model_modeller_tm_scores
from periscope.utils.utils import check_path, pkl_save, get_target_dataset

LOGGER = logging.getLogger(__name__)


def _save_plot_matrices(model: ContactMapEstimator, predictions, family=None):
    for target in predictions['logits']:
        data = {}
        ds = get_target_dataset(target) if family is None else family
        data_path = os.path.join(model.path, 'predictions', ds)
        check_path(data_path)
        target_path = os.path.join(data_path, target)
        check_path(target_path)
        dc = DataCreator(target, family=family)
        refs_contacts = dc.refs_contacts
        data['refs_contacts'] = refs_contacts
        # pd.DataFrame(refs_contacts).to_csv(os.path.join(target_path, 'refs_contacts.csv'))
        prediction = np.squeeze(predictions['logits'][target])
        data['prediction'] = prediction
        weights = np.squeeze(predictions['weights'][target])
        data['weights'] = weights
        # pd.DataFrame(prediction).to_csv(os.path.join(target_path, 'prediction.csv'))
        try:
            gt = dc.protein.cm
        except Exception:
            gt = None
        data['gt'] = gt
        # pd.DataFrame(gt).to_csv(os.path.join(target_path, 'gt.csv'))
        data['alignment'] = dc.templates_aln
        # data['evfold'] = dc.evfold
        data['ccmpred'] = dc.ccmpred
        data['templates'] = dc.k_reference_dm_test
        data['seqs'] = dc.seq_refs_ss_acc
        data['beff'] = dc.beff
        pkl_save(os.path.join(target_path, 'data.pkl'), data)
        upload_folder(target_path, target_path.split('Periscope/')[-1])


def parse_args():
    parser = ArgumentParser(description="Upload local folder to Google Drive")
    parser.add_argument('model', type=str, help='model name')
    parser.add_argument('-p','--proteins', nargs="+", help='target names')
    parser.add_argument('-ds', '--dataset', type=str, help='dataset', default=None)
    parser.add_argument('-f', '--family', type=str, help='family', default=None)
    parser.add_argument('-g', '--generate_data',
                        help='if true we generate the data in case it is missing', action="store_true")
    parser.add_argument('-d', '--three_d_model',
                        help='if true we generate 3d model', action="store_true")
    parser.add_argument('-t', '--template_free',
                        help='if true we do not require template for preidction', action="store_false")

    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model
    proteins = args.proteins
    dataset = args.dataset
    if dataset is not None:
        proteins = getattr(DATASETS_FULL, dataset)

    family = args.family
    get_d3_model = args.three_d_model

    require_template = args.template_free
    LOGGER.info(f'require template {require_template}')

    if args.generate_data:
        for p in proteins:
            data_creator = DataCreator(p, family=family)
            if not data_creator.has_msa:
                data_creator.generate_data()

    model = get_model_by_name(model_name)

    if family is not None:
        data_creator = DataCreator(proteins[0], family=family)
        for i in range(50):
            proteins = list(data_creator._parse_msa().keys())[(i*20):((i+1)*20)]
            predictions = model.predict(proteins=proteins, family=family, dataset=dataset)
            _save_plot_matrices(model, predictions, family=family)
        return

    predictions = model.predict(proteins=proteins, family=family, dataset=dataset)
    if dataset is not None:
        get_top_category_accuracy_np(predictions['logits'], model_path=model.path, model_name=model.name,
                                     dataset=dataset)

    _save_plot_matrices(model, predictions, family=family)
    if get_d3_model:
        for target in proteins:
            model_modeller_tm_scores(model_name, target)
    # for protein in trypsin:
    #     try:
    #         outfolder_p = os.path.join(outfolder_full, protein)
    #         check_path(outfolder_p)
    #         sequence = list(Protein(protein[0:4], protein[4]).str_seq)
    #         logits_df = pd.DataFrame(np.squeeze(logits[protein]), columns=sequence, index=sequence)
    #         logits_df.to_csv(os.path.join(outfolder_p, 'logits.csv'))
    #         pkl_save(os.path.join(outfolder_p, 'weights.pkl'), weights[protein])
    #     except KeyError:
    #         continue
    #
    # upload_folder(outfolder_full, os.path.join(outfolder, protein))


if __name__ == '__main__':
    main()
