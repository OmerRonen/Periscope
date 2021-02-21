import os
import shutil

from ..utils.constants import DATASETS_FULL, PATHS


def main():
    all_proteins = set(DATASETS_FULL.train) | set(DATASETS_FULL.eval) | set(DATASETS_FULL.pfam) | \
                   set(DATASETS_FULL.cameo) | set(DATASETS_FULL.cameo41) | set(DATASETS_FULL.membrane)
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
