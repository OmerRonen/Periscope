import os
import pytest
import numpy as np
from periscope.data_handler import ProteinDataHandler
from periscope.globals import (DATASETS, LOCAL, EVFOLD_PATH, PROTEIN_BOW_DIM, PROTEIN_BOW_DIM_PSSM,
                               PROTEIN_BOW_DIM_PSSM_SS, PROTEIN_BOW_DIM_SS)

targets = np.random.choice(DATASETS['eval'], 10)


@pytest.fixture(params=np.random.choice(DATASETS['eval'], 10))
def data_handler(request):
    return ProteinDataHandler(request.param, k=10)


class TestDataHandler:
    k = 10

    def test_pssm(self, data_handler):

        l = len(data_handler.protein.sequence)

        version = data_handler._MSA_VERSION

        pssm_file = os.path.join(data_handler.msa_data_path, 'pssm_bio_v%s.pkl' % version)
        if not pssm_file:
            return

        pssm = data_handler.pssm
        assert pssm.shape == (l, 20)

    def test_evfold(self, data_handler):

        version = min(data_handler._VERSION, data_handler._PLMC_VERSION)
        l = len(data_handler.protein.sequence)

        plmc_file = '%s_v%s.txt' % (data_handler.target, version)
        plmc_path = os.path.join(EVFOLD_PATH, plmc_file)

        if not os.path.isfile(plmc_path):
            return

        evfold_mat = data_handler.plmc_score
        assert evfold_mat.shape == (l, l)

    def test_reference_dm(self, data_handler):

        version = data_handler._STRUCTURES_VERSION
        reference = data_handler.closest_known_strucutre
        l = len(data_handler.protein.sequence)

        if reference is None:
            return

        dm_file = f'{reference}_pdb_v{version}.pkl'

        pdb_dist_mat_file = os.path.join(data_handler.msa_data_path, dm_file)
        if not os.path.isfile(pdb_dist_mat_file):
            return

        assert data_handler.reference_dm.shape == (l, l)

    def test_k_reference_dm(self, data_handler):

        k_ref_dm = data_handler.k_reference_dm
        k_ref_dm_conv = data_handler.k_reference_dm_conv
        l = len(data_handler.protein.sequence)

        if k_ref_dm is None:
            return
        assert k_ref_dm.shape == (l, l, 1, self.k)
        assert k_ref_dm_conv.shape == (l, l, self.k)

    def test_seq_refs(self, data_handler):

        seq_refs = data_handler.seq_refs
        l = len(data_handler.protein.sequence)

        if seq_refs is not None:
            assert seq_refs.shape == (l, PROTEIN_BOW_DIM, self.k)

        seq_refs_pssm = data_handler.seq_refs_pssm
        if seq_refs_pssm is not None:
            assert seq_refs_pssm.shape == (l, PROTEIN_BOW_DIM_PSSM, self.k)

        seq_refs_pssm_ss = data_handler.seq_refs_pssm_ss
        if seq_refs_pssm_ss is not None:
            assert seq_refs_pssm_ss.shape == (l, PROTEIN_BOW_DIM_PSSM_SS, self.k)

        seq_refs_ss = data_handler.seq_refs_ss
        if seq_refs_ss is not None:
            assert seq_refs_ss.shape == (l, PROTEIN_BOW_DIM_SS, self.k)

    def test_ss_vs_seq(self, data_handler):
        # we test that na's in the sequence are na's in the secondary structure
        if LOCAL:
            return
        refs = data_handler._get_k_closest_known_structures()

        aligned_ss = data_handler._get_aligned_ss()
        ref_seq_full = data_handler._get_seq_refs_full()
        msa = data_handler._parse_msa()
        target_seq_msa = str(msa[data_handler.target].seq)
        valid_inds = [i for i in range(len(target_seq_msa)) if target_seq_msa[i] != '-']
        for i, ref in enumerate(refs):
            ref_seq = np.array(msa[ref].seq)[valid_inds]
            if (ref_seq == '-').sum() == 0:
                continue
            assert aligned_ss[..., 7, i][ref_seq == '-'].mean() == 1
            assert ref_seq_full[..., 0, i][ref_seq == '-'].mean() == 1
