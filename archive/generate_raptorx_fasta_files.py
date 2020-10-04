import os

from .globals import periscope_path, MSA_DATA_PATH
from .utils_old import yaml_load


def _get_targets_fatsa(targets):
    txt = ''
    for t in targets:
        txt += _get_fasta_string(t)

    return txt


def _get_fasta_string(target):
    target_file = os.path.join(MSA_DATA_PATH, 'query', target + '.fasta')
    with open(target_file, 'r') as f:
        txt = f.readlines()

    return ''.join(txt)


def main():
    targets = list(yaml_load(os.path.join(periscope_path, 'data', 'seq_dist.yml')).keys())

    t1 = targets[0:50]
    t2 = targets[50:100]
    t3 = targets[100:]

    chunks = [t1, t2, t3]
    for i in range(len(chunks)):
        fasta_txt = _get_targets_fatsa(chunks[i])
        with open(os.path.join(periscope_path, 'data', f'raptor_chunk_{i + 1}.fasta'), 'w') as f:
            f.write(fasta_txt)


if __name__ == '__main__':
    main()
