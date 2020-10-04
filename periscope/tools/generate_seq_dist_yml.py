import os

from ..data.creator import DataCreator
from ..utils.constants import DATASETS, PATHS
from ..utils.utils import yaml_save


def main():
    seq_dist_data = {}
    test = set(DATASETS.pfam) | set(DATASETS.cameo) | set(DATASETS.cameo41) | set(DATASETS.membrane)
    for target in test:
        print(target)
        dh = DataCreator(target, 1)
        ref = dh.closest_reference
        if ref is None:
            continue
        seq_dist_data[target] = dh.get_plot_reference_data(ref)

    yaml_save(os.path.join(PATHS.periscope, 'data', 'seq_dist.yml'), seq_dist_data)


if __name__ == '__main__':
    main()
