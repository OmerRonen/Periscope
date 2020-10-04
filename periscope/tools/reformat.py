import subprocess
import os
import random

from ..utils.constants import PATHS
from ..utils.utils import get_target_path, check_path


def main():
    a2m_files = os.listdir(os.path.join(PATHS.hhblits, 'a2m'))
    random.shuffle(a2m_files)
    for file in a2m_files:
        target = file.split('.')[0]
        a2m_file = os.path.join(PATHS.hhblits, 'a2m', file)
        target_hhblits_path = os.path.join(get_target_path(target), 'hhblits')
        check_path(target_hhblits_path)
        fasta_file = os.path.join(target_hhblits_path, f'{target}_v2.fasta')
        a3m_file = os.path.join(target_hhblits_path, f'{target}.a3m')
        if not os.path.isfile(fasta_file):
            reformat = ['reformat.pl', a2m_file, fasta_file]
            subprocess.run(reformat)
        if not os.path.isfile(a3m_file):
            reformat = ['reformat.pl', a2m_file, a3m_file]
            subprocess.run(reformat)


if __name__ == '__main__':
    main()
