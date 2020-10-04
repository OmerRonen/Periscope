import os

from ..analysis.analyzer import get_model_predictions
from ..net.contact_map import get_model_by_name
from ..utils.constants import PATHS
from ..utils.protein import Protein
from ..utils.tm import _save_cns_data, _get_target_tm
from ..utils.utils import yaml_save


def main():
    ms = get_model_by_name('ms')
    dataset = 'cameo_sc'

    proteins = ["2n5uA",
                "4yo5L",
                "4z39B",
                "5an6C",
                "2n5dA",
                "2n24A",
                "2n8hA",
                "4zv0B",
                "2n5lA",
                "2myhA",
                "4xmqB",
                "2n11A",
                "2n4pA",
                "2n2cA",
                "4x0nB",
                "2n12A",
                "5awwG",
                "2n2eA",
                "2n32A",
                "2n5nA",
                "2mz0A",
                "2n8oA",
                "2mx7A",
                "2n8pA",
                "4yo3L"]

    predictions = get_model_predictions(model=ms, proteins=proteins, dataset=dataset)['logits']
    gts = {p: Protein(p[0:4], p[4]).cm for p in proteins}
    _save_cns_data(predictions=gts, dataset=dataset, model=ms)
    _save_cns_data(predictions=predictions, dataset=dataset + 'ms', model=ms)

    tms_gt = {}
    tms_ms = {}

    for target in proteins:
        try:
            gt_tm = _get_target_tm(target=target, model=ms, sswt=5, dataset=dataset)
            ms_tm = _get_target_tm(target=target, model=ms, sswt=5, dataset=dataset + 'ms')
        except IndexError:
            continue

        tms_gt[target] = gt_tm
        tms_ms[target] = ms_tm
        yaml_save(filename=os.path.join(PATHS.periscope, 'data', 'gt_tm.yaml'), data=tms_gt)
        yaml_save(filename=os.path.join(PATHS.periscope, 'data', 'ms_tm.yaml'), data=tms_ms)


if __name__ == '__main__':
    main()
