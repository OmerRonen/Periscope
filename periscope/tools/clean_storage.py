import os
import shutil

from ..utils.constants import DATASETS, PATHS


def main():
    all_proteins = set(DATASETS.train) | set(DATASETS.eval) | set(DATASETS.pfam) | \
                   set(DATASETS.cameo) | set(DATASETS.cameo41) | set(DATASETS.membrane)
    data = []
    folders = os.listdir(os.path.join(PATHS.data, 'proteins'))
    for f in folders:
        data += os.listdir(os.path.join(PATHS.data, 'proteins', f))

    proteins_to_remove = set(data).difference(all_proteins)
    for p in proteins_to_remove:
        try:
            shutil.rmtree(os.path.join(PATHS.data, 'proteins', p[1:3], p))
        except FileNotFoundError:
            pass


if __name__ == '__main__':
    main()
