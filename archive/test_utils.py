import numpy as np

from utils import generate_local_sequence_distance_mat, aa_dict


def test_generate_local_distance_mat():

    prot1 = np.array(['A', 'C', 'C'])
    prot2 = np.array(['A', 'C', '-'])

    def _numeric_prot(prot):
        num_prot = np.vectorize(aa_dict.__getitem__)(prot).astype(np.int32)

        return num_prot

    dist_mat = generate_local_sequence_distance_mat(_numeric_prot(prot1),
                                                    _numeric_prot(prot2),
                                                    local_range=2)

    assert (dist_mat == np.array([[0, 1 / 3, 1 / 3], [1 / 3, 2 / 3, 2 / 3],
                                  [1 / 3, 2 / 3, 2 / 3]])).all()
