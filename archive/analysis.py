import logging
import os
import re
import textwrap

import pandas as pd
import numpy as np
import matplotlib.pylab as plt

import matplotlib.patches as mpatches
import tensorflow as tf
from matplotlib.colors import ListedColormap

from .cm_predictor import ContactMapPredictor
from .data_handler import ProteinDataHandler
from .globals import MODELS_PATH, periscope_path, DATASETS, DRIVE_PATH
from .protein_net import get_top_category_accuracy, NetParams
from .utils import pkl_load, pkl_save, yaml_load
from sklearn.metrics import roc_curve

LOGGER = logging.getLogger(__name__)


def _check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


def _get_quant(l):
    return (l - 5) / (l - 1)


def _get_model_predictions(model):
    predicted_cms = list(model.get_predictions_generator())
    proteins = pkl_load(
        os.path.join(model.artifacts_path, 'predicted_proteins.pkl'))
    model_predictions = {'cm': {}, 'logits': {}}

    for protein, logits in dict(zip(proteins, predicted_cms)).items():
        contact_probs = logits[..., 0:1]
        l = contact_probs.shape[1]
        # top l/2 predictions
        quant = _get_quant(l)
        model_predictions['cm'][protein] = np.where(
            contact_probs >= np.quantile(
                contact_probs[np.triu_indices(contact_probs.shape[0])], quant),
            1, 0)

        model_predictions['logits'][protein] = contact_probs

    return model_predictions


def _get_seq_ref_msg(target):
    seq_info = yaml_load(os.path.join(periscope_path, 'data', 'seq_dist.yml'))
    return seq_info[target]['msg']


def _get_seq_ref_dist(target):
    seq_info = yaml_load(os.path.join(periscope_path, 'data', 'seq_dist.yml'))
    return seq_info[target]['sequence_distance']


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
    plt.savefig(os.path.join(periscope_path, 'data', 'figures', f'{model1}_{model2}_{dataset}_acc_diff.png'))


def _get_accuracy_per_seq_dist(model_name, dataset):
    model_path = os.path.join(DRIVE_PATH, model_name, 'artifacts', dataset)
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


def create_plot_data(target, model):
    data_handler = ProteinDataHandler(target, mode='ask')
    if not hasattr(data_handler, 'target_pdb_cm') or not hasattr(
            data_handler, 'modeller_cm'):
        return
    data_path = os.path.join(model.artifacts_path, 'data')
    _check_path(data_path)

    target_cm = pd.DataFrame(data_handler.target_pdb_cm)

    reference_cm = pd.DataFrame(data_handler.reference_cm)

    modeller_cm = pd.DataFrame(data_handler.modeller_cm)

    seq_dist = data_handler.sequence_distance_matrix
    close_ind = seq_dist < 6
    non_trivial_inds = seq_dist >= 6

    seq_dist[close_ind] = 0
    seq_dist[non_trivial_inds] = 1

    seq_dist_mask_met = pd.DataFrame(seq_dist)

    return {
        'mask': seq_dist_mask_met.values,
        'target': target_cm.values,
        'reference': reference_cm.values,
        'modeller': modeller_cm.values
    }


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
                         data_path,
                         figures_path,
                         target,
                         predicted_logits2=None,
                         model_name=None,
                         ref_name=None,
                         main_only=True):
    def _get_data(file_name):
        arr = pd.read_csv(os.path.join(data_path, '%s.csv' % file_name),
                          index_col=0).values
        return arr

    target_cm = _get_data('target_cm')
    aligned_reference = _get_data('aligned_reference')
    np.fill_diagonal(aligned_reference, 0)

    modeller = _get_data('modeller')

    fig_size = (12, 8) if main_only else (10, 10)
    plt.figure(figsize=fig_size, clear=True)

    if main_only:
        ax1 = plt.subplot(121)

    else:
        ax1 = plt.subplot(232)
        ax2 = plt.subplot(337)
        ax3 = plt.subplot(338)
        ax4 = plt.subplot(339)

    target_cm_unmasked = np.array(target_cm > 0, dtype=np.int32)

    def _get_accuracy(prediction_mat, target_cm):
        prediction_mat_triu = np.clip(prediction_mat[np.triu_indices(
            prediction_mat.shape[0])],
                                      a_min=0,
                                      a_max=1)
        target_cm_triu = target_cm[np.triu_indices(target_cm.shape[0])]
        true_predictions = np.logical_and(target_cm_triu > 0,
                                          prediction_mat_triu > 0)

        acc = np.round(true_predictions.sum() / prediction_mat_triu.sum(), 2)

        return acc

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
    quant = _get_quant(l*2)
    contact_threshold = np.quantile(predicted_logits[np.triu_indices(l)],
                                    quant)

    def _get_cm(logits):
        logits *= _get_mask(l)
        thres = np.quantile(logits[np.triu_indices(l)], quant)
        cm = np.where(logits >= thres, 1, 0)
        return cm

    def _color_predicted_cm(pred):

        pred_c = pred.copy()

        upper_triu_mask = np.zeros_like(pred_c, dtype=np.bool)
        upper_triu_mask[np.triu_indices(l)] = True

        native_contacts = target_cm > 0

        contact_predictions = np.logical_and(pred_c > 0, upper_triu_mask)

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

        np.fill_diagonal(pred_c, 0)

        return pred_c

    predicted_cm = _get_cm(predicted_logits)
    predicted_cm = _color_predicted_cm(predicted_cm)
    prediction_acc = _get_accuracy(predicted_cm, target_cm)

    predicted_cm[np.tril_indices(l)] = aligned_reference[np.tril_indices(l)]

    if predicted_logits2 is not None:
        contact_threshold2 = np.quantile(predicted_logits2[np.triu_indices(l)],
                                         quant)
        predicted_cm2 = _get_cm(predicted_logits2)
        predicted_cm2 = _color_predicted_cm(predicted_cm2)

        prediction_acc2 = _get_accuracy(predicted_cm2, target_cm)
        predicted_cm[np.tril_indices(l)] = predicted_cm2.T[np.tril_indices(l)]

    aligned_acc = _get_accuracy(aligned_reference, target_cm)
    modeller_acc = _get_accuracy(modeller, target_cm)

    if np.abs(modeller_acc - aligned_acc) > 0.2:
        print('Bad modeller %s' % target)

    def _get_plot_colors(arr):
        colors = {
            # np.nan: 'white',
            0: "black",
            1: "grey",
            2: "blue",
            3: "red",
            4: "lime",
            5: "yellow"
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

    ax1.matshow(predicted_cm, cmap=_get_plot_colors(predicted_cm))

    raw_msg = _get_seq_ref_msg(target)
    seq_dist = _get_seq_ref_dist(target)

    target_msg, ref_msg = raw_msg.split('\n')[0], raw_msg.split('\n')[1]
    n = 150
    target_msg_lines = [target_msg[i:i + n] for i in range(0, len(target_msg), n)]
    target_msg_lines[-1] += ' ' * (n - len(target_msg_lines[-1]))
    ref_msg_lines = [ref_msg[i:i + n] for i in range(0, len(ref_msg), n)]
    ref_msg_lines[-1] += ' ' * (n - len(ref_msg_lines[-1]))

    msg = [target_msg_lines[i] + '\n' + ref_msg_lines[i] for i in range(len(target_msg_lines))]

    split_sign = '  ' * (n // 2)

    msg = f"\n{split_sign}\n".join(msg)

    title = f'Prediction for {target}\n\nSequence distance to template: {seq_dist}\n{msg}'

    plt.suptitle(title, fontdict={'family': 'monospace'}, fontsize=6)

    ax1.plot([0, l - 1], [0, l - 1], 'k-', color='w')

    prediction_name = 'prediction' if model_name is None else model_name
    reference_name = 'structure' if ref_name is None else ref_name

    ax1.text(0.65,
             0.95,
             prediction_name,
             fontsize=8,
             bbox=dict(facecolor='red', alpha=0.2),
             color='white',
             transform=ax1.transAxes)
    ax1.text(0.05,
             0.05,
             reference_name,
             fontsize=8,
             bbox=dict(facecolor='red', alpha=0.2),
             color='white',
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
    colors = {1: 'grey', 2: 'blue', 3: 'red', 4: 'lime', 5: 'yellow'}
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
    ax2.matshow(col_ref, cmap=_get_plot_colors(col_ref))
    ax2.plot([0, predicted_cm.shape[0] - 1], [0, predicted_cm.shape[0] - 1],
             'k-',
             color='w')

    ax2.title.set_text('Aligned Reference ppv: %s' % aligned_acc)
    ax2.text(0.78,
             0.92,
             'reference',
             fontsize=6,
             bbox=dict(facecolor='red', alpha=0.2),
             color='white',
             transform=ax2.transAxes)
    ax2.text(0.05,
             0.05,
             'structure',
             fontsize=6,
             bbox=dict(facecolor='red', alpha=0.2),
             color='white',
             transform=ax2.transAxes)
    ax2.set_xticks([], [])
    ax2.set_yticks([], [])

    color_modeller = _color_predicted_cm(modeller)
    ax3.matshow(color_modeller, cmap=_get_plot_colors(color_modeller))
    ax3.plot([0, predicted_cm.shape[0] - 1], [0, predicted_cm.shape[0] - 1],
             'k-',
             color='w')

    ax3.title.set_text('Modeller ppv: %s' % modeller_acc)
    ax3.text(0.8,
             0.92,
             'modeller',
             fontsize=6,
             bbox=dict(facecolor='red', alpha=0.2),
             color='white',
             transform=ax3.transAxes)
    ax3.text(0.05,
             0.05,
             'structure',
             fontsize=6,
             bbox=dict(facecolor='red', alpha=0.2),
             color='white',
             transform=ax3.transAxes)
    ax3.set_xticks([], [])
    ax3.set_yticks([], [])

    fpr, tpr = _get_roc_data(predicted_logits, target_cm_unmasked)

    fpr_modeller, tpr_modeller = _get_fpr_tpr_(modeller, target_cm)
    fpr_reference, tpr_reference = _get_fpr_tpr_(aligned_reference, target_cm)
    fpr_method, tpr_method = _get_fpr_tpr_(_get_cm(predicted_logits), target_cm)

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
             color="darkviolet")
    ax4.annotate(model_name, (fpr_method, tpr_method), color='darkviolet')
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
    nested = {'short': None, 'medium': None, 'long': None}
    out_dict = {
        'target': nested.copy(),
        'other': nested.copy(),
        'closest': nested.copy()
    }
    target_imprv = {'s': [], 'm': [], 'l': []}
    for target in targets:
        dh = ProteinDataHandler(target, k=k)
        if len(dh.known_structures) < 3 or not hasattr(dh, 'k_reference_dm'):
            continue
        structures_arr = np.squeeze(dh.k_reference_dm)
        target_cm = dh.target_pdb_cm
        target_dm = dh.target_pdb_dm
        target_dm[target_dm == -1] = np.nan

        closest_structure = dh.reference_cm
        closest_structure_dm = dh.reference_dm
        closest_structure_dm[closest_structure_dm == -1] = np.nan

        seq_dist = dh.sequence_distance_matrix
        invalid_ind = seq_dist < 5
        target_cm[invalid_ind] = 0
        closest_structure[invalid_ind] = 0
        short_ind = np.logical_and(seq_dist > 5, seq_dist <= 11)
        medium_ind = np.logical_and(seq_dist > 11, seq_dist < 24)
        long_ind = seq_dist >= 24

        target_indices = target_cm == 1 if target_contact else target_cm == 0

        indices = np.logical_and(target_indices, closest_structure != 1)
        if all_contacts:
            indices = target_indices

        if nas:
            indices = np.logical_and(target_indices, closest_structure == -1)

        unidentified_contacts_ref_l = np.logical_and(indices, long_ind)
        unidentified_contacts_ref_m = np.logical_and(indices, medium_ind)
        unidentified_contacts_ref_s = np.logical_and(indices, short_ind)

        other_structues = min(len(dh.known_structures) - 2, k)
        other_structues_arr = structures_arr[...,
                              -1 * (other_structues + 1):-1]
        other_structues_arr[other_structues_arr == -1] = np.nan

        min_other = np.array(np.nanmin(other_structues_arr, axis=-1) < 8,
                             dtype=np.int)

        other_contact_correct = np.logical_and(min_other == 1, target_indices)

        improvement = np.array(other_contact_correct, dtype=np.int)

        if len(dh.known_structures) > 2 and imprv:
            l_s = max(np.array(target_cm[short_ind] == 1).sum(), 1)
            l_m = max(np.array(target_cm[medium_ind] == 1).sum(), 1)
            l_l = max(np.array(target_cm[long_ind] == 1).sum(), 1)
            l_val = improvement[unidentified_contacts_ref_l].sum() / l_l
            s_val = improvement[unidentified_contacts_ref_s].sum() / l_s
            m_val = improvement[unidentified_contacts_ref_m].sum() / l_m

            target_imprv['s'].append(np.round(s_val, 2))
            target_imprv['m'].append(np.round(m_val, 2))
            target_imprv['l'].append(np.round(l_val, 2))

        def _get_flatten_data(arr, inds):
            arr = np.expand_dims(arr, -1) if len(arr.shape) == 2 else arr
            extracted_ind = arr[inds, :].flatten()
            extracted_ind = extracted_ind[~np.isnan(extracted_ind)]
            return extracted_ind

        def _get_data_per_category(arr, long_inds, med_inds, short_inds):
            long_flat = _get_flatten_data(arr, long_inds)
            med_flat = _get_flatten_data(arr, med_inds)
            short_flat = _get_flatten_data(arr, short_inds)

            return {'short': short_flat, 'medium': med_flat, 'long': long_flat}

        target_per_category = _get_data_per_category(
            target_dm, unidentified_contacts_ref_l,
            unidentified_contacts_ref_m, unidentified_contacts_ref_s)
        other_per_category = _get_data_per_category(
            other_structues_arr, unidentified_contacts_ref_l,
            unidentified_contacts_ref_m, unidentified_contacts_ref_s)
        closest_per_category = _get_data_per_category(
            closest_structure_dm, unidentified_contacts_ref_l,
            unidentified_contacts_ref_m, unidentified_contacts_ref_s)

        for c in out_dict['target']:
            out_dict['target'][c] = target_per_category[
                c] if out_dict['target'][c] is None else np.concatenate(
                [out_dict['target'][c], target_per_category[c]])
            out_dict['other'][c] = other_per_category[
                c] if out_dict['other'][c] is None else np.concatenate(
                [out_dict['other'][c], other_per_category[c]])
            out_dict['closest'][c] = closest_per_category[
                c] if out_dict['closest'][c] is None else np.concatenate(
                [out_dict['closest'][c], closest_per_category[c]])
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

    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312)
    ax3 = plt.subplot(313)

    plt_dict = {'short': ax1, 'medium': ax2, 'long': ax3}

    bins = np.linspace(5, 15, 50)
    for c, ax in plt_dict.items():

        ax.hist(out_dict['target'][c],
                bins,
                label='target',
                color='blue',
                alpha=0.7)
        if not nas:
            ax.hist(out_dict['closest'][c],
                    bins,
                    label='closest structure',
                    color='red')
        ax.hist(out_dict['other'][c],
                bins,
                alpha=0.5,
                label='other structures',
                color='green')

        txt = 'other structure mean: %s\nclosest structure mean: %s\ntarget mean: %s' % (
            out_dict['other'][c].mean().round(2),
            out_dict['closest'][c].mean().round(2),
            out_dict['target'][c].mean().round(2))

        if np.isnan(out_dict['closest'][c].mean().round(2)):
            txt = 'other structure mean: %s\ntarget mean: %s' % (
                out_dict['other'][c].mean().round(2),
                out_dict['target'][c].mean().round(2))

        ax.text(0.01,
                0.83,
                txt,
                fontsize=8,
                bbox=dict(facecolor='blue', alpha=0.5),
                color='white',
                transform=ax.transAxes)

        ax.set_title(c)

        ax.legend(loc='upper right')

    plt.savefig(outfile)
    plt.close()


def plot_target_vs_reference(target, version):
    dh = ProteinDataHandler(target, structures_version=version)
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
    plt.savefig(os.path.join(periscope_path, 'data', f'cb_{target}_ref_v{version}.png'))
    plt.close()


def plot_target_reference(target, prediction, model, predicted_logits):
    data_path = os.path.join(model.artifacts_path, 'data')
    _check_path(data_path)

    figures_path = os.path.join(model.artifacts_path, 'figures')
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

    plot_data = create_plot_data(target, model)

    if plot_data is None:
        return

    mask = plot_data['mask']

    preds = np.squeeze(prediction)

    target_cm = _mask_cm(plot_data['target'], mask)
    aligned_reference = _mask_cm(plot_data['reference'], mask, target_cm)
    modeller = _mask_cm(plot_data['modeller'], mask, target_cm)
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
    target_path = os.path.join(model.artifacts_path, 'data', target)
    _check_path(target_path)

    pd.DataFrame(predicted_cm).to_csv(
        os.path.join(target_path, 'predicted_cm.csv'))
    pd.DataFrame(modeller).to_csv(os.path.join(target_path, 'modeller.csv'))
    pd.DataFrame(aligned_reference).to_csv(
        os.path.join(target_path, 'aligned_reference.csv'))
    pd.DataFrame(target_cm).to_csv(os.path.join(target_path, 'target_cm.csv'))
    pd.DataFrame(np.squeeze(predicted_logits)).to_csv(
        os.path.join(target_path, 'predicted_logits.csv'))

    plot_target_analysis(data_path=target_path,
                         figures_path=figures_path,
                         target=target,
                         predicted_logits=np.squeeze(predicted_logits))


def write_model_analysis(model_name):
    model_path = os.path.join(MODELS_PATH, model_name)
    params = NetParams(os.path.join(model_path, 'params.yml')).params
    model = ContactMapPredictor(params)
    predictions = _get_model_predictions(model)
    for target in predictions['cm'].keys():
        try:
            plot_target_reference(target=target,
                                  prediction=predictions['cm'][target],
                                  model=model,
                                  predicted_logits=predictions['logits'][target])
        except Exception:
            pass
    _get_top_category_accuracy_np(predictions['logits'], model, model_name)


def modeller_accuracy(dataset, model_name, version):
    model_full_name = f'modeller_{model_name}'
    prediction_data = {}

    targets = DATASETS[dataset]

    for target in targets:
        data_handler = ProteinDataHandler(target, mode='ask', structures_version=version)

        try:
            prediction = data_handler.get_average_modeller_cm(10, version=version)
            prediction = np.exp(-1 * prediction)
        except (TypeError, AttributeError):
            continue
        prediction_data[target] = (np.expand_dims(prediction, [0]), np.expand_dims(data_handler.target_pdb_cm, [0, 3]))

    model_path = os.path.join(periscope_path, 'models', model_full_name)
    _check_path(model_path)

    _save_accuracy(prediction_data,
                   model_path,
                   'modeller')


def reference_accuracy(dataset, model_name, version):
    model_full_name = f'reference_{model_name}'

    prediction_data = {}

    targets = DATASETS[dataset]

    for target in targets:
        data_handler = ProteinDataHandler(target, mode='ask', structures_version=version)

        try:
            prediction = np.exp(-1 * data_handler.reference_dm)
            prediction[data_handler.reference_dm == -1] = 0
        except (TypeError, AttributeError):
            continue
        prediction_data[target] = (np.expand_dims(prediction, [0]), np.expand_dims(data_handler.target_pdb_cm, [0, 3]))

    model_path = os.path.join(periscope_path, 'models', model_full_name)
    _check_path(model_path)

    _save_accuracy(prediction_data,
                   model_path,
                   'reference')


def _get_top_category_accuracy_np(logits, model, model_name):
    prediction_data = {}

    for target in logits.keys():
        data_handler = ProteinDataHandler(target, mode='ask')
        prediction_data[target] = (logits[target], data_handler.target_pdb_cm)

    _save_accuracy(prediction_data, model.artifacts_path, model_name)


def _save_accuracy(prediction_data, model_path, model_name):
    n_pred = [1, 2, 5, 10]

    categories = {
        'S': {n: {}
              for n in n_pred},
        'M': {n: {}
              for n in n_pred},
        'L': {n: {}
              for n in n_pred}
    }
    mode = tf.estimator.ModeKeys.PREDICT
    prediction_data_tf = {}
    for target, (pred_np, truth_np) in prediction_data.items():
        pred = tf.constant(pred_np, dtype=tf.float32)
        truth = tf.constant(truth_np, dtype=tf.float32)
        seq_len = tf.constant(np.max(pred_np.shape), dtype=tf.int32)
        prediction_data_tf[target] = (pred, truth, seq_len)

    with tf.Session():
        for category in categories:
            LOGGER.info('Evaluating category %s' % category)
            for top_np in categories[category]:
                LOGGER.info('Evaluating top L / %s' % top_np)

                for target in prediction_data_tf.keys():
                    top = tf.constant(top_np)
                    pred, truth, seq_len = prediction_data_tf[target]

                    acc = get_top_category_accuracy(category, top, pred, truth,
                                                    seq_len, mode).eval()

                    categories[category][top_np][target] = acc

    pkl_save(os.path.join(model_path, 'top_accuracy.pkl'), categories)

    keys_multiindex = [['Short', 'Medium', 'Long'],
                       ['Top L', 'Top L/2', 'Top L/5', 'Top L/10']]
    multi_index = pd.MultiIndex.from_product(keys_multiindex)
    data = []
    for k1 in categories:
        for k2 in categories[k1]:
            data.append(np.round(np.mean(list(categories[k1][k2].values())),
                                 2))

    model_acc = pd.Series(data, index=multi_index).round(2)

    all_models = pd.read_csv(os.path.join(periscope_path, 'data',
                                          'raptor_pfam.csv'),
                             index_col=[0, 1])

    accuracy = all_models.merge(model_acc.rename(model_name),
                                left_index=True,
                                right_index=True)

    accuracy.to_csv(os.path.join(model_path, 'accuracy.csv'))
