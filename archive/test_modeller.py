import pytest

from globals import MODELLER_PATH
from protein import Protein

from data_handler import ProteinDataHandler


@pytest.mark.parametrize(argnames='target', argvalues=['1ktgA', '5ptpA'])
def test_correct_init(target):
    protein, chain = target[0:4], target[4]
    pdb_prot = Protein(protein, chain)
    modeller_prot = Protein(protein, chain, pdb_path=MODELLER_PATH)
    d_h = ProteinDataHandler(target)
