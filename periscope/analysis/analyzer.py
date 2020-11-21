import logging
import os

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from scipy import stats

from ..data.creator import DataCreator
from ..data.seeker import DataSeeker
from ..net.contact_map import ContactMapEstimator, get_model_by_name
from ..utils.constants import DATASETS, PATHS
from ..utils.drive import upload_folder
from ..utils.protein import Protein
from ..utils.utils import (pkl_load, yaml_load, np_read_csv, get_dist_cat_mat, yaml_save, check_path, pkl_save,
                           get_target_dataset, get_data, get_raptor_logits)
from sklearn.metrics import roc_curve

LOGGER = logging.getLogger(__name__)


def _check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


def _get_quant(l, top=2):
    return ((l - 2 * top) - 1) / (l - 1)


def _get_average_prediction(model_names, dataset):
    model_predictions = {'cm': {}, 'logits': {}}

    models_dicts = {m: None for m in model_names}

    for model_name in model_names:
        model = get_model_by_name(model_name)
        predicted_cms = list(model.get_predictions_generator())
        proteins = pkl_load(os.path.join(model.artifacts_path, f'predicted_proteins_{dataset}.pkl'))
        models_dicts[model_name] = dict(zip(proteins, predicted_cms))

    for protein in proteins:
        models_logits = np.stack([m[protein][..., 0:1] for m in models_dicts.values()], axis=2)
        contact_probs = np.mean(models_logits, axis=2)

        l = contact_probs.shape[1]
        quant = np.quantile(contact_probs[np.triu_indices(contact_probs.shape[0])], _get_quant(l))
        model_predictions['cm'][protein] = np.where(contact_probs >= quant, 1, 0)
        model_predictions['logits'][protein] = contact_probs

    return model_predictions


def get_model_predictions(model: ContactMapEstimator, proteins=None, dataset=None):
    dataset = model.predict_data_manager.dataset if dataset is None else dataset
    predictions_path = os.path.join(model.path, 'predictions')
    check_path(predictions_path)
    prediction_file = os.path.join(predictions_path, f'{dataset}.pkl')
    if os.path.isfile(prediction_file) and proteins is None:
        return pkl_load(prediction_file)
    if proteins is not None:
        preds = list(model.get_custom_predictions_gen(proteins, dataset))
    else:
        preds = list(model.get_predictions_generator())

    proteins = pkl_load(
        os.path.join(model.artifacts_path, f'predicted_proteins_{dataset}.pkl'))
    model_predictions = {'cm': {}, 'logits': {}, 'weights':{}}

    for protein, pred in dict(zip(proteins, preds)).items():
        logits = pred['cm']
        weights = pred['weights']
        contact_probs = logits[..., 0:1]
        l = contact_probs.shape[1]
        # top l/2 predictions
        quant = _get_quant(l)
        model_predictions['cm'][protein] = np.where(
            contact_probs >= np.quantile(
                contact_probs[np.triu_indices(contact_probs.shape[0])], quant),
            1, 0)

        model_predictions['logits'][protein] = contact_probs
        model_predictions['weights'][protein] = weights

    if proteins is None:
        pkl_save(filename=prediction_file, data=model_predictions)

    return model_predictions


def _get_seq_ref_msg(target):
    try:
        seq_info = yaml_load(os.path.join(PATHS.periscope, 'data', 'seq_dist.yml'))
        return seq_info[target]['msg']
    except KeyError:
        return


def _get_seq_ref_dist(target):
    try:
        seq_info = yaml_load(os.path.join(PATHS.periscope, 'data', 'seq_dist.yml'))
        return seq_info[target]['sequence_distance']
    except KeyError:
        return


def _get_accuracy_per_seq_dist(model_name, dataset):
    model_path = os.path.join(PATHS.drive, model_name, 'artifacts', dataset)
    accuracy = yaml_load(os.path.join(model_path, 'top_accuracy.yaml'))

    accuracy_per_bin = {c: {1: [], 2: []} for c in ['S', 'M', 'L']}

    def _get_bin(seq_dist):

        if seq_dist < 0.6:
            return 1
        elif seq_dist <= 1:
            return 2

    targets = sorted(list(accuracy['S'][1].keys()))

    for cat, acc in accuracy.items():
        for target in targets:
            accuracy_per_bin[cat][_get_bin(_get_seq_ref_dist(target))].append(acc[1][target])
        # for bin in [1, 2]:
        #     accuracy_per_bin[cat][bin] = np.round(np.mean(accuracy_per_bin[cat][bin]), 2)

    return accuracy_per_bin


def _get_seq_dist_mat(target):
    data_seeker = DataSeeker(target, 1)

    if not hasattr(data_seeker, 'target_pdb_cm'):
        return

    seq_dist_mask = data_seeker.sequence_distance_matrix
    close_ind = seq_dist_mask < 6
    non_trivial_inds = seq_dist_mask >= 6

    seq_dist_mask[close_ind] = 0
    seq_dist_mask[non_trivial_inds] = 1

    return seq_dist_mask


def _get_roc_data(predicted_logits, target_cm):
    predicted_logits_triu = predicted_logits[np.triu_indices(
        predicted_logits.shape[0])]
    target_cm_triu = target_cm[np.triu_indices(target_cm.shape[0])]

    fpr, tpr, _ = roc_curve(target_cm_triu, predicted_logits_triu)
    return fpr, tpr


def dist(i, j):
    return np.abs(i - j)


def _get_mask(l):
    mask = np.fromfunction(dist, shape=(l, l))
    mask = np.where(mask > 5, 1, 0)
    return mask


def plot_target_analysis(predicted_logits,
                         figures_path,
                         target,
                         dataset,
                         predicted_logits2=None,
                         model_name=None,
                         ref_name=None,
                         main_only=False):
    def _get_data(file_name, path):
        fname = os.path.join(path, '%s.csv' % file_name)
        if not os.path.isfile(fname):
            return
        arr = np_read_csv(fname)
        return arr

    generic_data_path = os.path.join(PATHS.periscope, 'data', dataset)
    target_cm = Protein(target[0:4], target[4]).cm
    target_cm *= _get_mask(target_cm.shape[0])
    aligned_reference = _get_data(target, path=os.path.join(generic_data_path, 'reference'))
    aligned_reference = target_cm if aligned_reference is None else aligned_reference
    np.fill_diagonal(aligned_reference, 0)

    modeller = _get_data(target, path=os.path.join(generic_data_path, 'modeller'))
    modeller = np.zeros_like(target_cm) if modeller is None else modeller

    fig_size = (12, 8) if main_only else (10, 10)
    plt.figure(figsize=fig_size, clear=True)

    if main_only:
        ax1 = plt.subplot(121)

    else:
        ax1 = plt.subplot(232)
        ax3 = plt.subplot(234)
        ax4 = plt.subplot(236)

    target_cm_unmasked = np.array(target_cm > 0, dtype=np.int32)

    def _get_accuracy(prediction_mat, target_cm):
        prediction_mat_triu = np.clip(prediction_mat[np.triu_indices(
            prediction_mat.shape[0])],
                                      a_min=0,
                                      a_max=1)
        target_cm_triu = target_cm[np.triu_indices(target_cm.shape[0])]
        # true_predictions = np.logical_and(target_cm_triu > 0,
        #                                   prediction_mat_triu > 0)
        #
        # acc = np.round(true_predictions.sum() / prediction_mat_triu.sum(), 2)

        acc = np.round(np.mean(target_cm_triu[prediction_mat_triu > 0]), 2)
        return str(acc)

    def _get_fpr_tpr_(prediction_mat, target_cm):
        prediction_mat_triu = prediction_mat[np.triu_indices(
            prediction_mat.shape[0])]
        target_cm_triu = target_cm[np.triu_indices(target_cm.shape[0])]

        true_positive = np.logical_and(target_cm_triu > 0,
                                       prediction_mat_triu > 0)
        positive = target_cm_triu > 0
        tpr = true_positive.sum() / positive.sum()

        false_positive = np.logical_and(target_cm_triu == 0,
                                        prediction_mat_triu > 0)
        negative = target_cm_triu == 0
        fpr = false_positive.sum() / negative.sum()

        return fpr, tpr

    l = aligned_reference.shape[0]

    def _get_cm(logits, l):
        quant = _get_quant(l, 2)

        logits *= _get_mask(l)
        thres = np.quantile(logits[np.triu_indices(l)], quant)
        cm = np.where(logits >= thres, 1, 0)
        return cm, thres

    def _color_predicted_cm(pred):

        pred_c = pred.copy()

        upper_triu_mask = np.zeros_like(pred_c, dtype=np.bool)
        upper_triu_mask[np.triu_indices(l)] = True

        native_contacts = target_cm > 0

        contact_predictions = np.logical_and(pred_c, upper_triu_mask)

        correct_predictions = np.logical_and(target_cm > 0,
                                             contact_predictions)
        incorrect_predictions = np.logical_and(target_cm == 0,
                                               contact_predictions)

        not_in_ref = np.logical_and(aligned_reference == 0,
                                    correct_predictions)

        not_in_ref_or_modeller = np.logical_and(not_in_ref, modeller == 0)
        pred_c[native_contacts] = 1
        pred_c[correct_predictions] = 2
        pred_c[incorrect_predictions] = 3
        pred_c[not_in_ref] = 4
        pred_c[not_in_ref_or_modeller] = 5

        pred_c[_get_mask(l) == 0] = -1

        return pred_c

    predicted_cm, contact_threshold = _get_cm(predicted_logits, l)
    predicted_cm[np.tril_indices(l)] = aligned_reference[np.tril_indices(l)]

    predicted_cm = _color_predicted_cm(predicted_cm)
    prediction_acc = _get_accuracy(predicted_cm, target_cm)

    if predicted_logits2 is not None:
        predicted_cm2, contact_threshold2 = _get_cm(predicted_logits2, l)
        predicted_cm2 = _color_predicted_cm(predicted_cm2)

        prediction_acc2 = _get_accuracy(predicted_cm2, target_cm)
        predicted_cm[np.tril_indices(l)] = predicted_cm2.T[np.tril_indices(l)]

    # aligned_acc = _get_accuracy(aligned_reference, target_cm)

    def _get_plot_colors(arr):
        colors = {
            -1: 'lightgrey',
            0: "white",
            1: "grey",
            2: "darkblue",
            3: "lime",
            4: "red",
            5: "orange"
        }  # use hex colors here, if desired.
        colors_list = [colors[v] for v in np.unique(arr)]
        cmap = ListedColormap(colors_list)
        return cmap

    beyond_ref_contacts = np.array(predicted_cm[np.triu_indices(l)] > 3).sum() / np.array(
        predicted_cm[np.triu_indices(l)] > 1).sum()
    beyond_ref_contacts *= 100
    beyond_ref_contacts = np.round(beyond_ref_contacts, 2)

    beyond_ref_contacts_2 = np.array(predicted_cm[np.tril_indices(l)] > 3).sum() / np.array(
        predicted_cm[np.tril_indices(l)] > 1).sum()
    beyond_ref_contacts_2 *= 100
    beyond_ref_contacts_2 = np.round(beyond_ref_contacts_2, 2)

    ax1.matshow(predicted_cm, cmap=_get_plot_colors(predicted_cm), origin='lower')
    raw_msg = _get_seq_ref_msg(target)
    seq_dist = _get_seq_ref_dist(target)

    if raw_msg is not None:

        target_msg, ref_msg = raw_msg.split('\n')[0], raw_msg.split('\n')[1]
        n = 150
        target_msg_lines = [target_msg[i:i + n] for i in range(0, len(target_msg), n)]
        target_msg_lines[-1] += ' ' * (n - len(target_msg_lines[-1]))
        ref_msg_lines = [ref_msg[i:i + n] for i in range(0, len(ref_msg), n)]
        ref_msg_lines[-1] += ' ' * (n - len(ref_msg_lines[-1]))

        msg = [target_msg_lines[i] + '\n' + ref_msg_lines[i] for i in range(len(target_msg_lines))]

        split_sign = '  ' * (n // 2)

        msg = f"\n{split_sign}\n".join(msg)

    else:
        msg = ''

    title = f'Prediction for {target}\n\nSequence identity: {np.round(1 - seq_dist, 2)}\n\n{msg}'
    # title = f'Prediction for {target}'
    plt.suptitle(title, fontdict={'family': 'monospace'}, fontsize=6)

    prediction_name = 'prediction' if model_name is None else model_name
    reference_name = 'structure' if ref_name is None else ref_name

    ax1.text(0.05,
             0.90,
             reference_name,
             fontsize=8,
             color='black',
             transform=ax1.transAxes)
    ax1.text(0.70,
             0.05,
             prediction_name,
             fontsize=8,
             color='black',
             transform=ax1.transAxes)

    data = [[prediction_acc, np.round(contact_threshold, 2), beyond_ref_contacts]]
    table = pd.DataFrame(data, columns=['ppv', 'threshold', '%  new'], index=[model_name]).transpose()

    if ref_name is not None:
        data = [[prediction_acc, np.round(contact_threshold, 2), beyond_ref_contacts],
                [prediction_acc2, np.round(contact_threshold2, 2), beyond_ref_contacts_2]]
        table = pd.DataFrame(data, columns=['ppv', 'threshold', '%  new'], index=[model_name, ref_name]).transpose()
    cell_text = []
    for row in range(len(table)):
        cell_text.append(table.iloc[row])

    values = [1, 2, 3, 4, 5]
    colors = {1: 'grey', 2: 'darkblue', 3: 'lime', 4: 'red', 5: 'orange'}
    values_map = {
        1: 'native contact',
        2: 'contact identified',
        3: 'contact misidentified',
        4: 'not in reference',
        5: 'not in reference or modeller'
    }
    # create a patch (proxy artist) for every color
    patches = [
        mpatches.Patch(color=colors[i], label="{l}".format(l=values_map[i]))
        for i in values
    ]
    # put those patched as legend-handles into the legend
    ax1.legend(handles=patches,
               bbox_to_anchor=(1.05, 1),
               loc='upper left',
               borderaxespad=0.)
    ax1.set_xticks([], [])
    ax1.set_yticks([], [])
    the_table = ax1.table(cellText=cell_text, colLabels=table.columns, rowLabels=table.index)
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    the_table.scale(1, 2)

    if main_only:
        plt.savefig(os.path.join(figures_path, '%s_analysis.png' % target))
        # plt.show(False)
        plt.close('all')
        return
    col_ref = _color_predicted_cm(aligned_reference)
    # ax2.matshow(col_ref, cmap=_get_plot_colors(col_ref), origin='lower')
    #
    # ax2.title.set_text('Aligned Reference ppv: %s' % aligned_acc)
    # ax2.text(0.80,
    #          0.05,
    #          'reference',
    #          fontsize=6,
    #          color='black',
    #          transform=ax2.transAxes)
    # ax2.set_xticks([], [])
    # ax2.set_yticks([], [])

    modeller = _color_predicted_cm(modeller)
    modeller[np.tril_indices(l)] = col_ref.T[np.tril_indices(l)]
    modeller_acc = _get_accuracy(modeller, target_cm)

    ax3.matshow(modeller, cmap=_get_plot_colors(modeller), origin='lower')

    ax3.title.set_text('Modeller ppv: %s' % modeller_acc)

    ax3.text(0.80,
             0.05,
             'modeller',
             fontsize=6,
             color='black',
             transform=ax3.transAxes)
    ax3.text(0.05,
             0.9,
             'refs contacts',
             fontsize=6,
             color='black',
             transform=ax3.transAxes)
    ax3.set_xticks([], [])
    ax3.set_yticks([], [])

    fpr, tpr = _get_roc_data(predicted_logits, target_cm_unmasked)

    fpr_modeller, tpr_modeller = _get_fpr_tpr_(modeller - 1, target_cm)
    fpr_reference, tpr_reference = _get_fpr_tpr_(aligned_reference, target_cm)
    fpr_method, tpr_method = _get_fpr_tpr_(predicted_cm - 1, target_cm)

    ax4.plot(fpr, tpr, color='red')

    ax4.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax4.plot(fpr_modeller,
             tpr_modeller,
             marker='o',
             markersize=3,
             color="green")
    ax4.annotate('modeller', (fpr_modeller, tpr_modeller), color='green')
    ax4.plot(fpr_reference,
             tpr_reference,
             marker='o',
             markersize=3,
             color="blue")
    ax4.annotate('reference', (fpr_reference, tpr_reference), color='blue')
    ax4.plot(fpr_method,
             tpr_method,
             marker='o',
             markersize=3,
             color="darkred")
    ax4.annotate(model_name, (fpr_method, tpr_method), color='darkred')
    if ref_name is not None:
        fpr_2, tpr_2 = _get_roc_data(predicted_logits2, target_cm_unmasked)
        fpr_method_2, tpr_method_2 = _get_fpr_tpr_(predicted_cm2 - 1, target_cm)
        ax4.plot(fpr_2, tpr_2, color='orange')
        ax4.plot(fpr_method_2,
                 tpr_method_2,
                 marker='o',
                 markersize=3,
                 color="darkorange")
        ax4.annotate(ref_name, (fpr_method_2, tpr_method_2), color='darkorange')

    ax4.title.set_text('ROC Curve')
    ax4.set_xticks([], [])
    ax4.set_yticks([], [])

    ax4.set_xlabel('False Positive Rate')
    ax4.set_ylabel('True Positive Rate')

    plt.savefig(os.path.join(figures_path, '%s_analysis.png' % target))
    # plt.show(False)
    plt.close('all')


def investigate_structures_distribution(targets,
                                        all_contacts=False,
                                        nas=False,
                                        target_contact=True,
                                        imprv=False,
                                        k=5):
    out_dict = {
        'target': {'total': None},
        'other': {'total': None},
        'closest': {'total': None},
    }
    target_imprv = {'total': []}
    for target in targets:
        ds = DataSeeker(target, k)
        if ds.total_refs < 2 or not hasattr(ds, 'k_reference_dm_conv'):
            continue
        structures_arr = np.squeeze(ds.k_reference_dm_conv)
        target_cm = ds.target_pdb_cm
        target_dm = ds.target_pdb_dm
        target_dm[target_dm == -1] = np.nan

        closest_structure = ds.reference_cm
        closest_structure_dm = ds.reference_dm
        closest_structure_dm[closest_structure_dm == -1] = np.nan

        seq_dist = ds.sequence_distance_matrix
        invalid_ind = seq_dist < 5
        valid_ind = seq_dist >= 5
        target_cm[invalid_ind] = 0
        closest_structure[invalid_ind] = 0

        target_indices = target_cm == 1 if target_contact else target_cm == 0

        indices = np.logical_and(target_indices, closest_structure != 1)
        if all_contacts:
            indices = target_indices

        if nas:
            indices = np.logical_and(target_indices, closest_structure == -1)

        unidentified_contacts_ref = np.logical_and(indices, valid_ind)

        other_structues = min(ds.total_refs - 1, k)
        other_structues_arr = structures_arr[..., -1 * (other_structues + 1):-1]
        other_structues_arr[other_structues_arr == -1] = np.nan

        min_other = np.array(np.nanmin(other_structues_arr, axis=-1) < 8,
                             dtype=np.int)

        other_contact_correct = np.logical_and(min_other == 1, target_indices)

        improvement = np.array(other_contact_correct, dtype=np.int)

        if ds.total_refs > 2 and imprv:
            n_contacts = max(np.array(target_cm[valid_ind] == 1).sum(), 1)

            imprv_ratio = improvement[unidentified_contacts_ref].sum() / n_contacts

            target_imprv['total'].append(np.round(imprv_ratio, 2))

        def _get_flatten_data(arr, inds):
            arr = np.expand_dims(arr, -1) if len(arr.shape) == 2 else arr
            extracted_ind = arr[inds, :].flatten()
            extracted_ind = extracted_ind[~np.isnan(extracted_ind)]
            return extracted_ind

        target_hist = _get_flatten_data(target_dm, unidentified_contacts_ref)
        other_hist = _get_flatten_data(other_structues_arr, unidentified_contacts_ref)
        closest_hist = _get_flatten_data(closest_structure_dm, unidentified_contacts_ref)

        out_dict['target']['total'] = target_hist
        out_dict['other']['total'] = other_hist
        out_dict['closest']['total'] = closest_hist

    if imprv:
        return out_dict, target_imprv
    return out_dict


def _generate_structure_dict():
    nested = {'short': None, 'medium': None, 'long': None}
    out_dict = {
        'target': nested.copy(),
        'other': nested.copy(),
        'closest': nested.copy()
    }

    for c in out_dict['target']:
        out_dict['target'][c] = np.random.normal(8, 0.5, (100))
        out_dict['other'][c] = np.random.normal(8.3, 0.9, 90)
        out_dict['closest'][c] = np.random.normal(9, 0.1, (50))

    return out_dict


def plot_structures_distribution(out_dict, outfile, nas=False):
    plt.figure(figsize=(10, 10))

    ax = plt.subplot(111)
    total = 'total'

    def _plot_kernel(data, ax, label, color):
        if len(data) == 0:
            return
        data.sort()
        d = stats.gaussian_kde(dataset=data).evaluate(data)
        ax.plot(data, d, label=label, color=color, alpha=0.7)
        ax.fill_between(data, d, color=color, alpha=0.3)

    _plot_kernel(data=out_dict['target'][total], label='target', color='blue', ax=ax)
    if not nas:
        _plot_kernel(data=out_dict['closest'][total], label='closest', color='red', ax=ax)
    _plot_kernel(data=out_dict['other'][total], label='other structures', color='green', ax=ax)

    txt = 'other structure mean: %s\nclosest structure mean: %s\ntarget mean: %s' % (
        out_dict['other'][total].mean().round(2),
        out_dict['closest'][total].mean().round(2),
        out_dict['target'][total].mean().round(2))

    if np.isnan(out_dict['closest'][total].mean().round(2)):
        txt = 'other structure mean: %s\ntarget mean: %s' % (
            out_dict['other'][total].mean().round(2),
            out_dict['target'][total].mean().round(2))

    ax.text(0.01,
            0.92,
            txt,
            fontsize=12,
            bbox=dict(facecolor='gray', alpha=0.5),
            color='black',
            transform=ax.transAxes)

    ax.set_title(outfile.split('/')[0].split('.')[0].replace('_', " "))

    ax.legend(loc='upper right')

    plt.savefig(outfile)
    plt.close()


def plot_target_vs_reference(target, version):
    dh = DataSeeker(target, 1)
    ref = dh.reference_cm
    target_cm = dh.target_pdb_cm
    mask = np.array(dh.sequence_distance_matrix > 5, dtype=np.int32)
    mat = np.zeros_like(target_cm)
    l = target_cm.shape[0]
    mat[np.tril_indices(l)] = target_cm[np.tril_indices(l)]
    mat[np.triu_indices(l)] = ref[np.triu_indices(l)]
    mat *= mask
    target_cm[target_cm == -1] = 0
    ppv = target_cm[ref == 1].sum() / ref[ref == 1].sum()
    LOGGER.info(f"Full ppv for {target}, version {version} is : {ppv}")

    plt.matshow(mat, origin='lower')
    plt.savefig(os.path.join(PATHS.periscope, 'data', f'cb_{target}_ref_v{version}.png'))
    plt.close()


def plot_target_reference(target, prediction, model_path, predicted_logits, dataset, model_name):
    data_path = os.path.join(model_path, 'data')
    _check_path(data_path)

    figures_path = os.path.join(model_path, 'figures')
    _check_path(figures_path)

    def _mask_cm(cm, mask, target_masked=None):

        has_nans = np.array(cm == -1).sum()

        if has_nans > 0:
            cm[cm == -1] = np.nan

        cm_masked = np.array(np.nan_to_num(cm, nan=0) * mask, dtype=np.int)

        if target_masked is not None:
            l = target_masked.shape[0]
            lower_tri_indices = np.tril_indices(l)

            cm_masked[lower_tri_indices] = target_masked[lower_tri_indices]
            np.fill_diagonal(cm_masked, -1)

        return cm_masked

    mask = _get_seq_dist_mat(target)

    if mask is None:
        return

    preds = np.squeeze(prediction)

    data_creator = DataCreator(target, n_refs=1)

    target_cm = _mask_cm(data_creator.target_pdb_cm, mask)

    generic_data_path = os.path.join(PATHS.periscope, 'data', dataset)
    _check_path(generic_data_path)

    _check_path(os.path.join(generic_data_path, 'reference'))
    _check_path(os.path.join(generic_data_path, 'modeller'))
    _check_path(os.path.join(generic_data_path, 'target'))

    ref_file = os.path.join(generic_data_path, 'reference', f'{target}.csv')
    aligned_reference = _mask_cm(data_creator.refs_contacts, mask, target_cm)
    pd.DataFrame(aligned_reference).to_csv(ref_file)
    # if not os.path.isfile(ref_file):
    #     aligned_reference = _mask_cm(data_creator.refs_contacts, mask, target_cm)
    #     pd.DataFrame(aligned_reference).to_csv(ref_file)
    # else:
    #     aligned_reference = np_read_csv(ref_file)
    modeller_file = os.path.join(generic_data_path, 'modeller', f'{target}.csv')
    modeller = _mask_cm(data_creator.modeller_cm, mask, target_cm)
    pd.DataFrame(modeller).to_csv(modeller_file)
    # if not os.path.isfile(modeller_file):
    #     modeller = _mask_cm(data_creator.modeller_cm, mask, target_cm)
    #     pd.DataFrame(modeller).to_csv(modeller_file)
    # else:
    #     modeller = np_read_csv(modeller_file)
    target_file = os.path.join(generic_data_path, 'target', f'{target}.csv')
    pd.DataFrame(target_cm).to_csv(target_file)

    # if not os.path.isfile(target_file):
    #     pd.DataFrame(target_cm).to_csv(target_file)

    predicted_cm = _mask_cm(preds, mask, target_cm)
    upper_triu_mask = np.zeros_like(predicted_cm, dtype=np.bool)
    upper_triu_mask[np.triu_indices(predicted_cm.shape[0])] = True

    contact_predictions = np.logical_and(predicted_cm > 0, upper_triu_mask)

    correct_predictions = np.logical_and(target_cm > 0,
                                         contact_predictions)
    incorrect_predictions = np.logical_and(target_cm == 0,
                                           contact_predictions)

    not_in_ref = np.logical_and(aligned_reference == 0,
                                correct_predictions)

    not_in_ref_or_modeller = np.logical_and(not_in_ref, modeller == 0)
    predicted_cm[correct_predictions] = 2
    predicted_cm[incorrect_predictions] = 3
    predicted_cm[not_in_ref] = 4
    predicted_cm[not_in_ref_or_modeller] = 5

    # we save those file so that we can recreate the plot locally
    target_path = os.path.join(model_path, 'data')
    _check_path(target_path)

    pd.DataFrame(predicted_cm).to_csv(
        os.path.join(target_path, f'{target}_cm.csv'))
    pd.DataFrame(predicted_logits).to_csv(
        os.path.join(target_path, f'{target}_logits.csv'))

    plot_target_analysis(figures_path=figures_path,
                         target=target,
                         dataset=dataset,
                         predicted_logits=np.squeeze(predicted_logits),
                         model_name=model_name)


def _save_plot_matrices(model: ContactMapEstimator, predictions):
    for target in predictions['logits']:
        data = {}
        ds = get_target_dataset(target)
        if ds is None:
            LOGGER.warning(f'Problem with {target}, does not belong to any dataset')
            continue
        data_path = os.path.join(model.path, 'predictions', ds)
        check_path(data_path)
        target_path = os.path.join(data_path, target)
        check_path(target_path)
        dc = DataCreator(target)
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
        data['evfold'] = dc.evfold
        data['ccmpred'] = dc.ccmpred
        data['templates'] = dc.k_reference_dm_test
        data['seqs'] = dc.seq_refs_ss_acc
        data['beff'] = dc.beff
        pkl_save(os.path.join(target_path,'data.pkl'), data)
        upload_folder(target_path, target_path.split('Periscope/')[-1])


def save_model_predictions(model: ContactMapEstimator, protein, outfile):
    predictions = get_model_predictions(model, proteins=[protein])['logits']
    sequence = list(Protein(protein[0:4], protein[4]).str_seq)
    pd.DataFrame(np.squeeze(predictions[protein]), columns=sequence, index=sequence).to_csv(outfile)


def save_model_analysis(model: ContactMapEstimator, proteins=None):
    predictions = get_model_predictions(model, proteins=proteins)
    if proteins is None:
        _get_top_category_accuracy_np(predictions['logits'], model.path, model.name, model.predict_data_manager.dataset)

    _save_plot_matrices(model, predictions)


def write_model_analysis(model, model_path, model_name, dataset, models=None, plot=True):
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    predictions = get_model_predictions(model) if models is None else _get_average_prediction(models, dataset)
    _get_top_category_accuracy_np(predictions['logits'], model_path, model_name, dataset)
    if plot:
        for target in predictions['cm'].keys():
            try:
                plot_target_reference(target=target,
                                      prediction=predictions['cm'][target],
                                      model_path=model_path,
                                      predicted_logits=np.squeeze(predictions['logits'][target]),
                                      dataset=dataset,
                                      model_name=model_name)
            except Exception:
                pass


def modeller_accuracy(dataset):
    model_full_name = f'modeller'
    prediction_data = {}

    targets = getattr(DATASETS, dataset)

    for target in targets:
        dc = DataCreator(target, 1)

        try:
            prediction = dc.get_average_modeller_dm(n_structures=4)
            prediction = np.exp(-1 * prediction)
        except (TypeError, AttributeError):
            LOGGER.info(f'Skipping {target}')
            continue
        prediction_data[target] = (np.expand_dims(prediction, [0]), np.expand_dims(dc.target_pdb_cm, [0, 3]))

    model_path = os.path.join(PATHS.periscope, 'models', model_full_name, dataset)
    _check_path(model_path)

    _save_accuracy(prediction_data,
                   model_path,
                   'modeller',
                   dataset)


def reference_accuracy(dataset):
    model_full_name = f'reference'

    prediction_data = {}

    targets = getattr(DATASETS, dataset)

    for target in targets:
        data_handler = DataSeeker(target, 1)

        try:
            prediction = np.exp(-1 * data_handler.reference_dm)
            prediction[data_handler.reference_dm == -1] = 0
        except (TypeError, AttributeError):
            continue
        prediction_data[target] = (np.expand_dims(prediction, [0]), np.expand_dims(data_handler.target_pdb_cm, [0, 3]))

    model_path = os.path.join(PATHS.periscope, 'models', model_full_name, dataset)
    _check_path(model_path)

    _save_accuracy(prediction_data,
                   model_path,
                   'reference',
                   dataset)


def plot_models_accuracy_per_dist(model1, model2, dataset):
    acc1 = _get_accuracy_per_seq_dist(model1, dataset)
    acc2 = _get_accuracy_per_seq_dist(model2, dataset)

    fig = plt.figure()
    ax = plt.axes()

    # plt.hold(True)

    def _setBoxColors(bp):
        plt.setp(bp['boxes'][0], color='blue')
        plt.setp(bp['caps'][0], color='blue')
        plt.setp(bp['caps'][1], color='blue')
        plt.setp(bp['whiskers'][0], color='blue')
        plt.setp(bp['whiskers'][1], color='blue')
        plt.setp(bp['fliers'][0], color='blue')
        plt.setp(bp['medians'][0], color='blue')

        plt.setp(bp['boxes'][1], color='red')
        plt.setp(bp['caps'][2], color='red')
        plt.setp(bp['caps'][3], color='red')
        plt.setp(bp['whiskers'][2], color='red')
        plt.setp(bp['whiskers'][3], color='red')
        plt.setp(bp['fliers'][1], color='red')
        plt.setp(bp['medians'][1], color='red')

    ind_pos = 1
    for cat in acc1:
        diff_close_ref = np.array(acc1[cat][1]) - np.array(acc2[cat][1])
        diff_far_ref = np.array(acc1[cat][2]) - np.array(acc2[cat][2])

        d = [diff_close_ref * 100, diff_far_ref * 100]
        bp = plt.boxplot(d, positions=[ind_pos, ind_pos + 1], widths=0.6)
        _setBoxColors(bp)
        ind_pos += 4

    ax.set_xticklabels(list(acc1.keys()))
    ax.set_xticks([1.5, 5.5, 9.5])

    hB, = plt.plot([1, 1], 'b-')
    hR, = plt.plot([1, 1], 'r-')
    plt.legend((hB, hR), ("sequence identity > 0.4", "sequence identity < 0.4"))
    hB.set_visible(False)
    hR.set_visible(False)
    plt.ylim(-10, 25)
    plt.xlabel('distance category')
    plt.ylabel('ppv % diff')
    plt.title(f'Diff in % of native contacts identfied\n{model1} minus {model2}')
    plt.savefig(os.path.join(PATHS.periscope, 'data', 'figures', f'{model1}_{model2}_{dataset}_acc_diff.png'))


def _get_accuracy_per_seq_dist(model_name, dataset):
    model_path = os.path.join(PATHS.drive, model_name, 'artifacts', dataset)
    accuracy = yaml_load(os.path.join(model_path, 'top_accuracy.yaml'))

    accuracy_per_bin = {c: {1: [], 2: []} for c in ['S', 'M', 'L']}

    def _get_bin(seq_dist):

        if seq_dist < 0.6:
            return 1
        elif seq_dist <= 1:
            return 2

    targets = sorted(list(accuracy['S'][1].keys()))

    for cat, acc in accuracy.items():
        for target in targets:
            accuracy_per_bin[cat][_get_bin(_get_seq_ref_dist(target))].append(acc[1][target])
        # for bin in [1, 2]:
        #     accuracy_per_bin[cat][bin] = np.round(np.mean(accuracy_per_bin[cat][bin]), 2)

    return accuracy_per_bin


def _get_top_category_accuracy_np(logits, model_path, model_name, dataset):
    logits = {k: v for k, v in logits.items() if v is not None}
    prediction_data = {}

    for target in logits.keys():
        cm = Protein(target[0:4], target[4]).cm
        prediction_data[target] = (logits[target], cm)

    _save_accuracy(prediction_data, model_path, model_name, dataset)


def calculate_accuracy(logits, gt):
    n_pred = [1, 2, 5, 10]

    categories = {
        'S': {n: {}
              for n in n_pred},
        'M': {n: {}
              for n in n_pred},
        'L': {n: {}
              for n in n_pred}
    }

    cat_values_map = {'S': 1, 'M': 2, "L": 3}
    l = logits.shape[0]

    seq_dist_mat = get_dist_cat_mat(l)
    for category in categories:
        inds = seq_dist_mat == cat_values_map[category]

        logits_cat = logits[inds]
        gt_cat = gt[inds]
        df_comb = pd.DataFrame({'gt': gt_cat, 'pred': logits_cat})
        sorted_gt = df_comb.sort_values('pred', ascending=False).loc[:, 'gt'].values

        for top_np in categories[category]:
            n_preds = int(np.ceil(l / top_np))
            categories[category][top_np] = float(sorted_gt[0:n_preds].mean())

    return categories


def ds_accuracy(model, dataset):
    prediction_data = {}
    ds_path = os.path.join(PATHS.models, model.name, 'predictions', dataset)
    for target in os.listdir(ds_path):
        if target not in getattr(DATASETS, dataset):
            continue
        data = get_data(model.name, target)
        raptor_logits = get_raptor_logits(target)
        if raptor_logits is None:
            continue
        logits = data['prediction']
        gt = data['gt']
        prediction_data[target] = (logits, gt, raptor_logits)
    LOGGER.info(f'Number of predictions for {model.name} dataset {dataset} is {len(prediction_data)}')
    n_pred = [1, 2, 5, 10]

    categories = {
        'S': {n: {}
              for n in n_pred},
        'M': {n: {}
              for n in n_pred},
        'L': {n: {}
              for n in n_pred}
    }
    categories_raptor = {
        'S': {n: {}
              for n in n_pred},
        'M': {n: {}
              for n in n_pred},
        'L': {n: {}
              for n in n_pred}
    }
    cat_values_map = {'S': 1, 'M': 2, "L": 3}

    for category in categories:
        for target in prediction_data.keys():
            logits = np.squeeze(prediction_data[target][0])
            gt = np.squeeze(prediction_data[target][1])
            logits_raptor = np.squeeze(prediction_data[target][2])

            if gt.shape != logits.shape:
                continue
            l = logits.shape[0]

            seq_dist_mat = get_dist_cat_mat(l)
            inds = seq_dist_mat == cat_values_map[category]

            logits_cat = logits[inds]
            gt_cat = gt[inds]
            logits_raptor_cat = logits_raptor[inds]
            sorted_gt = pd.DataFrame({'gt': gt_cat, 'pred': logits_cat}).sort_values('pred', ascending=False).loc[:,
                        'gt'].values
            sorted_gt_raptor = pd.DataFrame({'gt': gt_cat, 'pred': logits_raptor_cat}).sort_values('pred', ascending=False).loc[:,
                        'gt'].values
            for top_np in categories[category]:
                n_preds = int(np.ceil(l / top_np))
                categories[category][top_np][target] = float(sorted_gt[0:n_preds].mean())
                categories_raptor[category][top_np][target] = float(sorted_gt_raptor[0:n_preds].mean())

    keys_multiindex = [['Short', 'Medium', 'Long'],
                       ['Top L', 'Top L/2', 'Top L/5', 'Top L/10']]
    multi_index = pd.MultiIndex.from_product(keys_multiindex)
    data = []
    data_raptor = []
    for k1 in categories:
        for k2 in categories[k1]:
            data.append(np.round(np.mean(list(categories[k1][k2].values())),
                                 2))
            data_raptor.append(np.round(np.mean(list(categories_raptor[k1][k2].values())),
                                 2))

    model_acc = pd.Series(data, index=multi_index).round(2)
    raptor_acc = pd.Series(data_raptor, index=multi_index).round(2)

    raptor_file = os.path.join(PATHS.data,
                               f'raptor_{dataset}.csv')
    if not os.path.isfile(raptor_file):
        raptor_file = os.path.join(PATHS.data,
                                   f'raptor_pfam.csv')

    all_models = pd.read_csv(raptor_file, index_col=[0, 1])

    accuracy = all_models.merge(model_acc.rename(model.name),
                                left_index=True,
                                right_index=True)

    accuracy['RaptorX_new'] = raptor_acc

    return accuracy


def _save_accuracy(prediction_data, model_path, model_name, dataset):
    n_pred = [1, 2, 5, 10]

    categories = {
        'S': {n: {}
              for n in n_pred},
        'M': {n: {}
              for n in n_pred},
        'L': {n: {}
              for n in n_pred}
    }

    cat_values_map = {'S': 1, 'M': 2, "L": 3}

    for category in categories:
        for target in prediction_data.keys():
            logits = np.squeeze(prediction_data[target][0])
            gt = np.squeeze(prediction_data[target][1])
            if gt.shape != logits.shape:
                continue
            l = logits.shape[0]

            seq_dist_mat = get_dist_cat_mat(l)
            inds = seq_dist_mat == cat_values_map[category]

            logits_cat = logits[inds]
            gt_cat = gt[inds]
            sorted_gt = pd.DataFrame({'gt': gt_cat, 'pred': logits_cat}).sort_values('pred', ascending=False).loc[:,
                        'gt'].values

            for top_np in categories[category]:
                n_preds = int(np.ceil(l / top_np))
                categories[category][top_np][target] = float(sorted_gt[0:n_preds].mean())

    yaml_save(filename=os.path.join(model_path, 'top_accuracy.yaml'), data=categories)

    keys_multiindex = [['Short', 'Medium', 'Long'],
                       ['Top L', 'Top L/2', 'Top L/5', 'Top L/10']]
    multi_index = pd.MultiIndex.from_product(keys_multiindex)
    data = []
    for k1 in categories:
        for k2 in categories[k1]:
            data.append(np.round(np.mean(list(categories[k1][k2].values())),
                                 2))

    model_acc = pd.Series(data, index=multi_index).round(2)

    raptor_file = os.path.join(PATHS.data,
                               f'raptor_{dataset}.csv')
    if not os.path.isfile(raptor_file):
        raptor_file = os.path.join(PATHS.data,
                                   f'raptor_pfam.csv')

    all_models = pd.read_csv(raptor_file, index_col=[0, 1])

    accuracy = all_models.merge(model_acc.rename(model_name),
                                left_index=True,
                                right_index=True)

    accuracy.to_csv(os.path.join(model_path, 'accuracy.csv'))
