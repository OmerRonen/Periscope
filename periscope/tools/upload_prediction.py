import logging
import os

import numpy as np

from argparse import ArgumentParser

from periscope.analysis.analyzer import get_model_predictions
from periscope.data.creator import DataCreator
from periscope.net.contact_map import get_model_by_name, ContactMapEstimator
from periscope.utils.drive import upload_folder
from periscope.utils.utils import check_path, pkl_save, get_target_dataset

logging.getLogger().setLevel(logging.CRITICAL)


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
        gt = dc.protein.cm
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
    # parser.add_argument('proteins', nargs="+", help='target names')
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model
    # proteins = args.proteins
    protein = '1mbqA'
    data_creator = DataCreator(protein, family='trypsin')
    trypsin = list(set(data_creator._parse_msa().keys()).difference(set(data_creator.aligner.get_ref_map().keys())))[0:100]#list(data_creator.aligner.get_ref_map().values()) + [protein]
    model = get_model_by_name(model_name)
    predictions = get_model_predictions(model, proteins=trypsin, family='trypsin')
    _save_plot_matrices(model, predictions, family="trypsin")

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
