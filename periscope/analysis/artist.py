import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pylab as plt
import matplotlib.patches as mpatches

from matplotlib.colors import ListedColormap
from sklearn.metrics import roc_curve

from ..utils.constants import PATHS, DATASETS
from ..utils.protein import Protein
from ..utils.utils import get_target_dataset, check_path, pkl_load, get_data, get_raptor_logits





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


def get_cm(logits, l):
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


def _get_default_colors(cmap='Set1'):
    if cmap == 'Set1':
        cm_ = plt.cm.Set1
    elif cmap == 'Set2':
        cm_ = plt.cm.Set2
    default_colors = np.array(pd.DataFrame(np.array([cm_(x / (20 - 1)) for x in range(20)])[:, :-1]).drop_duplicates())
    return default_colors


def _get_color(weights, slope=4,
               offset=-0.3, cmap='Set1'
               ):
    import matplotlib
    default_colors = _get_default_colors(cmap=cmap)
    ncolors = len(default_colors)
    default_colors_hsv = matplotlib.colors.rgb_to_hsv(default_colors)
    index = np.argmax(weights, axis=-1) % ncolors
    value_multiplier = np.maximum(np.minimum(slope * (weights.max(-1) + offset), 1), 0)
    colors = default_colors_hsv[index, :]
    colors[:, -1] *= value_multiplier
    colors = matplotlib.colors.hsv_to_rgb(colors)
    return colors


def plot_weights(model_name, target,
                 value_slope=5, value_offset=-0.1, proba_contact_cutoff=0.5, background_alpha=0.1,
                 cmap='Set1', family=None):
    data = get_data(model_name, target, family)
    weights = data['weights']
    predictions = data['prediction']
    L = weights.shape[0]
    aln = data['alignment']
    template_names = {(10 - i): aln[i].id for i in range(1, len(aln))}
    template_names[10] = 'ccmpred'
    relevant_templates = np.unique(np.argmax(weights, axis=-1))
    has_above_threshold = (weights[:, :, relevant_templates].max(0).max(0) > - value_offset)
    relevant_templates = relevant_templates[has_above_threshold]

    ntemplates = len(relevant_templates)
    template_names = [template_names[i] for i in relevant_templates]
    weights = weights[:, :, relevant_templates]

    if ntemplates > 0:
        color = _get_color(weights.reshape([L ** 2, -1]), slope=value_slope, offset=value_offset, cmap=cmap).reshape(
            [L, L, 3])
    else:
        color = np.zeros([L, L, 3])
    position = np.array(np.meshgrid(np.arange(L), np.arange(L)))
    contacts = (predictions > proba_contact_cutoff)

    fig, ax = plt.subplots(figsize=(8, 8))

    marker_size = np.sqrt(fig.get_size_inches()[0] * fig.dpi / L)

    ax.set_facecolor((0., 0., 0., 0.1))
    plt.scatter(position[0][contacts], L - position[1][contacts], c=color[contacts], marker='s',
                s=marker_size

                )

    if ntemplates > 0:
        default_colors = _get_default_colors(cmap=cmap)
        ncolors = len(default_colors)
        for i in range(ntemplates):
            plt.scatter([-1], [-1], c=default_colors[i % ncolors], marker='s', s=marker_size, label=template_names[i])
        ax.legend(fontsize=12, markerscale=5, frameon=True, loc='upper left');
    ax.set_ylim(ax.get_ylim()[::-1])  # invert the axis

    ax.set_xticks([])
    ax.set_yticks([])
    dataset = get_target_dataset(target, family)
    # prediction_path = os.path.join(PATHS.drive, model_name, 'predictions', dataset, target)
    fig_path = os.path.join(PATHS.drive, model_name, 'artifacts', dataset, target)
    fig_name = os.path.join(fig_path, f'weights.png')
    plt.savefig(fig_name)


def plot_weights_2(model_name, target, value_slope=5, value_offset=-0.2, proba_contact_cutoff=0.5,
                   background_alpha=0.1):
    data = get_data(model_name, target)
    weights = data['weights']
    predictions = data['prediction']
    L = weights.shape[0]
    color = _get_color(weights.reshape([L ** 2, -1]), slope=value_slope, offset=value_offset).reshape([L, L, 3])
    position = np.array(np.meshgrid(np.arange(L), np.arange(L)))
    contacts = (predictions > proba_contact_cutoff)

    fig, ax = plt.subplots(figsize=(8, 8))

    marker_size = np.sqrt(fig.get_size_inches()[0] * fig.dpi / L)
    print(marker_size)
    ax.set_facecolor((0., 0., 0., 0.1))
    plt.scatter(position[0][contacts], L - position[1][contacts], c=color[contacts], marker='s',
                s=marker_size

                )
    ax.set_ylim(ax.get_ylim()[::-1])  # invert the axis

    ax.set_xticks([])
    ax.set_yticks([])
    dataset = get_target_dataset(target)
    # prediction_path = os.path.join(PATHS.drive, model_name, 'predictions', dataset, target)
    fig_path = os.path.join(PATHS.drive, model_name, 'artifacts', dataset, target)
    fig_name = os.path.join(fig_path, f'weights.png')
    plt.savefig(fig_name)


def plot_weights_old(model_name, target, order=1):
    data = get_data(model_name, target)
    weights = data['weights']
    sort_w = np.argsort(weights, axis=-1)
    max_w = sort_w[..., -1 * order]
    sort_w_value = np.sort(weights, axis=-1)
    w_alpha = sort_w_value[..., -1 * order]
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
    if order == 1:
        plt.title('Largest Weight')
    if order == 2:
        plt.title('Second Largest Weight')
    ax1.set_xticks([], [])
    ax1.set_yticks([], [])
    dataset = get_target_dataset(target)
    # prediction_path = os.path.join(PATHS.drive, model_name, 'predictions', dataset, target)
    fig_path = os.path.join(PATHS.drive, model_name, 'artifacts', dataset, target)
    fig_name = os.path.join(fig_path, f'weights_{order}.png')
    plt.savefig(fig_name)


def evaluate_pred_vs_ref(model_name, target, family):
    data = get_data(model_name, target, family)
    dataset = get_target_dataset(target, family)
    fig_path = os.path.join(PATHS.drive, model_name, 'artifacts', dataset, target)
    fig_name = os.path.join(fig_path, 'pred_ref.png')
    check_path(fig_path)
    gt = data['gt']
    l = gt.shape[0]
    pred_model, _ = get_cm(data['prediction'], l)
    refs_contact = data['refs_contacts']
    labels = list(Protein(target[0:4], target[4]).str_seq)
    evaluation_plot(pred1=pred_model, pred2=refs_contact, gt=gt, fig_name=fig_name, pred2_name='References',
                    labels=labels)


def evaluate_pred_vs_modeller(model_name, target, family):
    data = get_data(model_name, target, family)
    dataset = get_target_dataset(target, family)
    fig_path = os.path.join(PATHS.drive, model_name, 'artifacts', dataset, target)
    fig_name = os.path.join(fig_path, 'pred_mod.png')
    check_path(fig_path)
    gt = data['gt']
    l = gt.shape[0]
    pred_model, _ = get_cm(data['prediction'], l)
    mod_ds = get_target_dataset(target) if get_target_dataset(target) is not None else family
    modeller = _read_csv_np(os.path.join(PATHS.data,mod_ds , 'modeller', f'{target}.csv'))
    if modeller is None:
        return
    labels = list(Protein(target[0:4], target[4]).str_seq)

    evaluation_plot(pred1=pred_model, pred2=modeller, gt=gt, fig_name=fig_name, pred2_name='Modeller', labels=labels)


def evaluate_pred_vs_raptor(model_name, target, family):
    data = get_data(model_name, target, family)
    dataset = get_target_dataset(target, family)
    fig_path = os.path.join(PATHS.drive, model_name, 'artifacts', dataset, target)
    fig_name = os.path.join(fig_path, 'pred_raptor.png')
    check_path(fig_path)
    gt = data['gt']
    l = gt.shape[0]
    pred_model, _ = get_cm(data['prediction'], l)
    raptor_logits = get_raptor_logits(target)
    if raptor_logits is None:
        return
    pred_raptor, _ = get_cm(raptor_logits, l)
    labels = list(Protein(target[0:4], target[4]).str_seq)

    evaluation_plot(pred1=pred_model, pred2=pred_raptor, gt=gt, fig_name=fig_name, pred2_name='Raptor', labels=labels)


def evaluate_pred_vs_evo(model_name, target, family):
    for evo in ['ccmpred']:
        data = get_data(model_name, target, family)
        dataset = get_target_dataset(target, family)
        fig_path = os.path.join(PATHS.drive, model_name, 'artifacts', dataset, target)
        fig_name = os.path.join(fig_path, f'pred_{evo}.png')
        check_path(fig_path)
        gt = data['gt']
        l = gt.shape[0]
        pred_model, _ = get_cm(data['prediction'], l)
        pred_evo, _ = get_cm(data[evo], l)
        labels = list(Protein(target[0:4], target[4]).str_seq)

        evaluation_plot(pred1=pred_model, pred2=pred_evo, gt=gt, fig_name=fig_name, pred2_name=evo, labels=labels)


def evaluate_pred_roc(model_name, target, family):
    data = get_data(model_name, target, family)
    dataset = get_target_dataset(target, family)
    fig_path = os.path.join(PATHS.drive, model_name, 'artifacts', dataset, target)
    fig_name = os.path.join(fig_path, 'roc.png')
    check_path(fig_path)
    gt = data['gt']

    l = gt.shape[0]
    mask = _get_mask(l)
    pred_logits = data['prediction'] * mask
    refs = data['refs_contacts'] * mask if data['refs_contacts'] is not None else np.zeros_like(mask)
    r_logits = get_raptor_logits(target)
    raptor_logits = r_logits * mask if r_logits is not None else np.zeros_like(mask)

    try:
        ds = get_target_dataset(target)
        ds = ds if ds is not None else dataset
        mod= _read_csv_np(os.path.join(PATHS.data,ds, 'modeller', f'{target}.csv'))
        modeller = mod * mask if mod is not None else np.zeros_like(mask)

    except ValueError as e:
        modeller = np.zeros_like(raptor_logits)
    gt *= mask
    _plot_roc(pred_logits, gt, modeller, refs, fig_name=fig_name, pred2_logits=raptor_logits, method2_name='RaptorX')


def _plot_roc(pred_logits, gt, modeller, refs, pred2_logits=None, fig_name=None, method2_name=None):
    fig_size = (12, 8)
    plt.figure(figsize=fig_size, clear=True)

    ax4 = plt.subplot(111)
    l = gt.shape[0]
    gt[gt == -1] = 0
    predicted_cm1, _ = get_cm(pred_logits, l)
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
    if pred2_logits is not None:
        fpr_2, tpr_2 = _get_roc_data(pred2_logits, gt)
        predicted_cm2, _ = get_cm(pred2_logits, l)
        fpr_method2, tpr_method2 = _get_fpr_tpr_(predicted_cm2, gt)
        ax4.plot(fpr_2, tpr_2, color='orange')
        ax4.plot(fpr_method2,
                 tpr_method2,
                 marker='o',
                 markersize=3,
                 color="orange")
        ax4.annotate(method2_name, (fpr_method2, tpr_method2), color='orange')
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


def make_art(model_name, target, family=None):
    data = get_data(model_name, target, family)
    if data is None:
        return
    evaluate_pred_vs_ref(model_name, target, family)
    evaluate_pred_vs_evo(model_name, target, family)
    evaluate_pred_vs_raptor(model_name, target, family)

    evaluate_pred_roc(model_name, target, family)
    evaluate_pred_vs_modeller(model_name, target, family)
    plot_weights(model_name, target, family=family)
    # plot_weights_old(model_name, target)
    #
    # try:
    #     plot_weights_old(model_name, target, 2)
    # except Exception:
    #     pass
