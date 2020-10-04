import os

from .data_handler import ProteinDataHandler
from .utils_old import yaml_save
from .globals import DATASETS, periscope_path


def main():
    seq_dist_data = {}
    pfam = DATASETS['pfam']
    for target in pfam:
        dh = ProteinDataHandler(target)
        seq_dist_data[target] = dh.get_plot_reference_data(dh.closest_known_strucutre)

    yaml_save(os.path.join(periscope_path, 'data', 'seq_dist.yml'), seq_dist_data)


if __name__ == '__main__':
    main()
