import pytest

import numpy as np

from periscope.globals import DATASETS
from periscope.protein import Protein


@pytest.fixture(params=np.random.choice(DATASETS['train'], 300))
def protein(request):
    return Protein(request.param[0:4], request.param[4])


class TestProtein:
    def test_matching_dimentions(self, protein):
        angles = protein.angels
        distance_matrix = protein.dm
        sequence = protein.sequence
        secondary_structure = protein.secondary_structure

        assert angles.shape[0] == distance_matrix.shape[0] == len(sequence) == secondary_structure.shape[0]
