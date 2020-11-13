import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.patches as mpatches

from matplotlib.colors import ListedColormap
from sklearn.metrics import roc_curve

from ..utils.constants import PATHS, DATASETS
from ..utils.protein import Protein
from ..utils.utils import get_target_dataset, check_path, pkl_load, get_data


def _read_csv_np(filename):
    if not os.path.isfile(filename):
        return
    return pd.read_csv(filename, index_col=0).values


def _get_quant(l, top=2):
    return ((l - 2 * top) - 1) / (l - 1)


def dist(i, j):
    return np.abs(i - j)


def _get_mask(l):
    mask = np.fromfunction(dist, shape=(l, l))
    mask = np.where(mask > 5, 1, 0)
    return mask


def _get_cm(logits, l):
    quant = _get_quant(l, 2)

    logits *= _get_mask(l)
    thres = np.quantile(logits[np.triu_indices(l)], quant)
    cm = np.where(logits >= thres, 1, 0)
    return cm, thres


def _get_accuracy(prediction, gt):
    prediction_mat_triu = np.clip(prediction[np.triu_indices(
        prediction.shape[0])],
                                  a_min=0,
                                  a_max=1)
    target_cm_triu = gt[np.triu_indices(gt.shape[0])]

    acc = np.round(np.mean(target_cm_triu[prediction_mat_triu > 0]), 2)
    return str(acc)


def _get_plot_colors(arr):
    colors = {
        -1: 'lightgrey',
        0: "white",
        1: "grey",
        2: "lime",
        3: "red",
        4: "grey",
        5: "orange"
    }  # use hex colors here, if desired.
    colors_list = [colors[v] for v in sorted(np.unique(arr))]
    cmap = ListedColormap(colors_list, name='colors', N=None)
    return cmap


def _color_pred_vs_gt(prediction, gt):
    native_contacts = gt == 1
    predicted_contacts = prediction == 1
    true_positive = np.logical_and(native_contacts, predicted_contacts)
    false_positive = np.logical_and(np.invert(native_contacts), predicted_contacts)
    false_negative = np.logical_and(native_contacts, np.invert(predicted_contacts))
    prediction[true_positive] = 2
    prediction[false_positive] = 3
    prediction[false_negative] = 4
    return prediction


def _get_legend_info(plot_matrix):
    values = set(np.unique(plot_matrix)).difference({0})
    colors = {1: 'grey', 2: 'lime', 3: 'red', 4: 'grey'}
    values_map = {
        1: 'native contact',
        2: 'contact identified (true positive)',
        3: 'contact misidentified (false positive)',
        4: 'contact overlooked (false negative)',
    }
    # create a patch (proxy artist) for every color
    patches = [
        mpatches.Patch(color=colors[i], label="{l}".format(l=values_map[i]))
        for i in values
    ]
    return patches


def _process_prediction(pred, gt):
    l = gt.shape[0]
    mask = _get_mask(l)
    if pred is None or pred.shape != gt.shape:
        return gt * mask
    p = pred.copy()

    pred1 = _color_pred_vs_gt(p, gt)
    pred1 *= mask
    return pred1


def evaluation_plot(pred1, gt, pred2_name, pred2=None, fig_name=None, labels=None):
    l = gt.shape[0]
    pred_1 = _process_prediction(pred1, gt)
    pred_2 = _process_prediction(pred2, gt)

    plot_matrix = np.zeros_like(gt)
    lower_t = np.tril_indices(l)
    upper_t = np.triu_indices(l)
    plot_matrix[lower_t] = pred_2[lower_t]
    plot_matrix[upper_t] = pred_1[upper_t]

    _plot_mat(plot_matrix, fig_name=fig_name, pred2_name=pred2_name, labels=labels)


def _plot_mat(plot_matrix, pred2_name, fig_name=None, labels=None):
    fig_size = (21, 14)
    plt.figure(figsize=fig_size, clear=True)

    ax1 = plt.subplot(121)
    l = plot_matrix.shape[0]

    cmap = _get_plot_colors(plot_matrix)
    legend_info = _get_legend_info(plot_matrix)

    if 1 not in np.unique(plot_matrix):
        plot_matrix[plot_matrix > 0] = plot_matrix[plot_matrix > 0] - 1

    ax1.matshow(plot_matrix, cmap=cmap, origin='lower')

    ax1.legend(handles=legend_info, loc='upper center', bbox_to_anchor=(0.5, 1.3),
               ncol=1, fancybox=True, shadow=True, prop={'size': 18})
    ax1.plot([0, l], [0, l], c=".3")

    ax1.set_xticks([], [])
    ax1.set_yticks([], [])
    ax1.text(-0.05,
             0.80,
             pred2_name,
             fontsize=16,
             color='blue',
             transform=ax1.transAxes,
             rotation=90)
    ax1.text(1.05,
             0.05,
             'Periscope',
             fontsize=16,
             color='blue',
             transform=ax1.transAxes,
             rotation=90)
    # if labels is not None:
    #     ax1.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False, color='blue')
    #
    #     plt.xticks(range(len(labels)), labels, size=4, color='blue')
    #     ax1.tick_params(axis=u'both', which=u'both', length=0, bottom=True, top=False)
    #
    #     plt.yticks(range(len(labels)), labels, size=4, color='blue')
    if fig_name is not None:
        plt.savefig(fig_name)
    plt.close()


def _get_roc_data(prediction, gt):
    predicted_logits_triu = prediction[np.triu_indices(
        prediction.shape[0])]
    target_cm_triu = gt[np.triu_indices(gt.shape[0])]

    fpr, tpr, _ = roc_curve(target_cm_triu, predicted_logits_triu)
    return fpr, tpr


def _get_fpr_tpr_(prediction, gt):
    prediction_mat_triu = prediction[np.triu_indices(
        prediction.shape[0])]
    target_cm_triu = gt[np.triu_indices(gt.shape[0])]

    true_positive = np.logical_and(target_cm_triu > 0,
                                   prediction_mat_triu > 0)
    positive = target_cm_triu > 0
    tpr = true_positive.sum() / positive.sum()

    false_positive = np.logical_and(target_cm_triu == 0,
                                    prediction_mat_triu > 0)
    negative = target_cm_triu == 0
    fpr = false_positive.sum() / negative.sum()

    return fpr, tpr


def target_analysis(model_name, target):
    evaluate_pred_roc(model_name, target)
    evaluate_pred_vs_ref(model_name, target)


def dataset_analysis(model_name, dataset):
    for target in getattr(DATASETS, dataset):
        try:
            target_analysis(model_name, target)
        except Exception:
            pass


def plot_weights(model_name, target, order=1):
    data = get_data(model_name, target)
    weights = data['weights']
    sort_w = np.argsort(weights, axis=-1)
    max_w = sort_w[..., -1 * order]
    sort_w_value = np.sort(weights, axis=-1)
    w_alpha = sort_w_value[..., -1*order]
    fig_size = (21, 14)
    plt.figure(figsize=fig_size, clear=True)

    ax1 = plt.subplot(121)
    im = ax1.matshow(max_w, interpolation='none', alpha=w_alpha, origin='lower')
    values = np.unique(max_w.ravel())
    is_evo_net = weights.shape[-1] == 2
    ks = {10: "ccmpred", 11: "evfold"} if not is_evo_net else {0: "ccmpred", 1: "evfold"}
    if not is_evo_net:
        aln = data['alignment']
        n_refs = len(aln) - 1
        j = 10 - n_refs
        for t in aln[1:]:
            ks[j] = t.id
            j += 1

    # get the colors of the values, according to the
    # colormap used by imshow

    colors = {value: im.cmap(im.norm(value)) for value in values}
    # create a patch (proxy artist) for every color
    patches = [mpatches.Patch(color=colors[i], label="{l}".format(l=ks[i])) for i in values]
    # put those patched as legend-handles into the legend
    ax1.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.5, 0.8),
               ncol=1, fancybox=True, shadow=True, prop={'size': 18})
    if order==1:
        plt.title('Largest Weight')
    if order==2:
        plt.title('Second Largest Weight')
    ax1.set_xticks([], [])
    ax1.set_yticks([], [])
    dataset = get_target_dataset(target)
    # prediction_path = os.path.join(PATHS.drive, model_name, 'predictions', dataset, target)
    fig_path = os.path.join(PATHS.drive, model_name, 'artifacts', dataset, target)
    fig_name = os.path.join(fig_path, f'weights_{order}.png')
    plt.savefig(fig_name)


def evaluate_pred_vs_ref(model_name, target):
    data = get_data(model_name, target)
    dataset = get_target_dataset(target)
    fig_path = os.path.join(PATHS.drive, model_name, 'artifacts', dataset, target)
    fig_name = os.path.join(fig_path, 'pred_ref.png')
    check_path(fig_path)
    gt = data['gt']
    l = gt.shape[0]
    pred_model, _ = _get_cm(data['prediction'], l)
    refs_contact = data['refs_contacts']
    labels = list(Protein(target[0:4], target[4]).str_seq)
    evaluation_plot(pred1=pred_model, pred2=refs_contact, gt=gt, fig_name=fig_name, pred2_name='References',
                    labels=labels)


def evaluate_pred_vs_modeller(model_name, target):
    data = get_data(model_name, target)
    dataset = get_target_dataset(target)
    fig_path = os.path.join(PATHS.drive, model_name, 'artifacts', dataset, target)
    fig_name = os.path.join(fig_path, 'pred_mod.png')
    check_path(fig_path)
    gt = data['gt']
    l = gt.shape[0]
    pred_model, _ = _get_cm(data['prediction'], l)
    modeller = _read_csv_np(os.path.join(PATHS.data, dataset, 'modeller', f'{target}.csv'))
    if modeller is None:
        return
    labels = list(Protein(target[0:4], target[4]).str_seq)

    evaluation_plot(pred1=pred_model, pred2=modeller, gt=gt, fig_name=fig_name, pred2_name='Modeller', labels=labels)


def _get_raptor_logits(target):
    drive = '/Users/omerronen/Google Drive (omerronen10@gmail.com)/Periscope'
    raptor_path = os.path.join(drive, 'raptorx')
    target_file = os.path.join(raptor_path, f'{target}.predictedDistMatrix.pkl')
    if not os.path.isfile(target_file):
        return
    raptor_data = pkl_load(target_file)
    return raptor_data[3]['CbCb']


def evaluate_pred_vs_raptor(model_name, target):
    data = get_data(model_name, target)
    dataset = get_target_dataset(target)
    fig_path = os.path.join(PATHS.drive, model_name, 'artifacts', dataset, target)
    fig_name = os.path.join(fig_path, 'pred_raptor.png')
    check_path(fig_path)
    gt = data['gt']
    l = gt.shape[0]
    pred_model, _ = _get_cm(data['prediction'], l)
    raptor_logits = _get_raptor_logits(target)
    if raptor_logits is None:
        return
    pred_raptor, _ = _get_cm(raptor_logits, l)
    labels = list(Protein(target[0:4], target[4]).str_seq)

    evaluation_plot(pred1=pred_model, pred2=pred_raptor, gt=gt, fig_name=fig_name, pred2_name='Raptor', labels=labels)


def evaluate_pred_vs_evo(model_name, target):
    for evo in ['ccmpred', 'evfold']:
        data = get_data(model_name, target)
        dataset = get_target_dataset(target)
        fig_path = os.path.join(PATHS.drive, model_name, 'artifacts', dataset, target)
        fig_name = os.path.join(fig_path, f'pred_{evo}.png')
        check_path(fig_path)
        gt = data['gt']
        l = gt.shape[0]
        pred_model, _ = _get_cm(data['prediction'], l)
        pred_evo, _ = _get_cm(data[evo], l)
        labels = list(Protein(target[0:4], target[4]).str_seq)

        evaluation_plot(pred1=pred_model, pred2=pred_evo, gt=gt, fig_name=fig_name, pred2_name=evo, labels=labels)


def evaluate_pred_roc(model_name, target):
    data = get_data(model_name, target)
    dataset = get_target_dataset(target)
    fig_path = os.path.join(PATHS.drive, model_name, 'artifacts', dataset, target)
    fig_name = os.path.join(fig_path, 'roc.png')
    check_path(fig_path)
    gt = data['gt']

    l = gt.shape[0]
    mask = _get_mask(l)
    pred_logits = data['prediction'] * mask
    refs = data['refs_contacts'] * mask
    modeller = _read_csv_np(os.path.join(PATHS.data, dataset, 'modeller', f'{target}.csv')) * mask
    gt *= mask
    _plot_roc(pred_logits, gt, modeller, refs, fig_name=fig_name)


def _plot_roc(pred_logits, gt, modeller, refs, pred2_logits=None, fig_name=None):
    fig_size = (12, 8)
    plt.figure(figsize=fig_size, clear=True)

    ax4 = plt.subplot(111)
    l = gt.shape[0]
    gt[gt == -1] = 0
    predicted_cm1, _ = _get_cm(pred_logits, l)
    fpr, tpr = _get_roc_data(pred_logits, gt)

    fpr_modeller, tpr_modeller = _get_fpr_tpr_(modeller, gt)
    fpr_reference, tpr_reference = _get_fpr_tpr_(refs, gt)
    fpr_method, tpr_method = _get_fpr_tpr_(predicted_cm1, gt)

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
    ax4.annotate("Periscope", (fpr_method, tpr_method), color='darkred')
    ax4.title.set_text('ROC Curve')
    ax4.set_xticks([], [])
    ax4.set_yticks([], [])

    ax4.set_xlabel('False Positive Rate')
    ax4.set_ylabel('True Positive Rate')
    if fig_name is not None:
        plt.savefig(fig_name)
    plt.close()
    # if pred2_logits is not None:
    #     fpr_2, tpr_2 = _get_roc_data(predicted_logits2, target_cm_unmasked)
    #     fpr_method_2, tpr_method_2 = _get_fpr_tpr_(predicted_cm2 - 1, gt)
    #     ax4.plot(fpr_2, tpr_2, color='orange')
    #     ax4.plot(fpr_method_2,
    #              tpr_method_2,
    #              marker='o',
    #              markersize=3,
    #              color="darkorange")
    #     ax4.annotate(ref_name, (fpr_method_2, tpr_method_2), color='darkorange')


def make_art(model_name, target):
    evaluate_pred_vs_ref(model_name, target)
    evaluate_pred_roc(model_name, target)
    evaluate_pred_vs_modeller(model_name, target)
    evaluate_pred_vs_raptor(model_name, target)
    evaluate_pred_vs_evo(model_name, target)
    plot_weights(model_name, target)
    try:
        plot_weights(model_name, target, 2)
    except Exception:
        pass