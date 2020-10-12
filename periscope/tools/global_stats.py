import os

from ..data.creator import DataCreator
from ..utils.constants import DATASETS, PATHS
from ..utils.utils import get_aln_fasta, yaml_save, yaml_load


def get_stats():
    training = set(DATASETS.eval) | set(DATASETS.train)
    stats_file = os.path.join(PATHS.data, 'stats_msa.yaml')
    stats = {'msa': {}, 'tempaltes': {}} if not os.path.isfile(stats_file) else yaml_load(stats_file)

    for t in training:
        if t in stats['msa']:
            continue
        msa_file = get_aln_fasta(t)
        if not os.path.isfile(msa_file):
            continue
        dc = DataCreator(t)
        if not dc.has_refs:
            continue
        stats['msa'][t] = dc.msa_length
        stats['tempaltes'][t] = len(dc.sorted_structures)

        yaml_save(filename=os.path.join(PATHS.data, 'stats_msa.yaml'), data=stats)


if __name__ == '__main__':
    get_stats()
