import os
from collections import namedtuple
import pandas as pd
import yaml


def yaml_load(filename):
    with open(filename, 'r') as stream:
        data_loaded = yaml.load(stream)

    return data_loaded


C_MAP_PRED = 'c_map_prediction'
UPPER_TRIANGULAR_MSE_LOSS = 'upper_triangular_mse_loss'
UPPER_TRIANGULAR_CE_LOSS = 'upper_triangular_cross_entropy_loss'
CONTACT_MAP_PH = 'contact_map_ph'

Features = namedtuple(
    'Features',
    'target_pdb_cm target_pdb_dm plmc_score target_outer_angels pssm '
    'reference_cm reference_dm reference_dm_local_dist reference_std k_reference_dm '
    'k_reference_dm_local_dist k_reference_dm_conv k_reference_dm_conv_local_dist '
    'seq_target seq_refs seq_target_pssm seq_refs_pssm ccmpred seq_target_pssm_ss seq_refs_pssm_ss seq_target_ss '
    'seq_refs_ss')

FEATURES = Features(
    target_pdb_cm='target_pdb_cm',
    target_pdb_dm='target_pdb_dm',
    plmc_score='plmc_score',
    target_outer_angels='target_outer_angels',
    pssm='pssm',
    reference_cm='reference_cm',
    reference_dm='reference_dm',
    reference_dm_local_dist='reference_dm_local_dist',
    reference_std='reference_std',
    k_reference_dm='k_reference_dm',
    k_reference_dm_local_dist='k_reference_dm_local_dist',
    k_reference_dm_conv='k_reference_dm_conv',
    k_reference_dm_conv_local_dist='k_reference_dm_conv_local_dist',
    seq_target='seq_target',
    seq_refs='seq_refs',
    seq_target_pssm='seq_target_pssm',
    seq_refs_pssm='seq_refs_pssm',
    ccmpred='ccmpred',
    seq_target_pssm_ss='seq_target_pssm_ss',
    seq_refs_pssm_ss='seq_refs_pssm_ss',
    seq_refs_ss='seq_refs_ss',
    seq_target_ss='seq_target_ss'
)
PROTEIN_BOW_DIM = 22
PROTEIN_BOW_DIM_SS = 30
PROTEIN_BOW_DIM_PSSM = 42
PROTEIN_BOW_DIM_PSSM_SS = 50

DRIVE_PATH = '/Users/omerronen/Google Drive (omerronen10@gmail.com)/Periscope/models'

FEATURES_DIMS = {
    'target_pdb_cm': (None, None, 1),
    'target_pdb_dm': (None, None, 1),
    'plmc_score': (None, None, 1),
    'ccmpred': (None, None, 1),
    'target_outer_angels': (None, None, 3),
    'pssm': (None, None, 44),
    'reference_cm': (None, None, 1),
    'reference_dm': (None, None, 1),
    'reference_dm_local_dist': (None, None, 2),
    'reference_std': (None, None, 2),
    'k_reference_dm': (None, None, 1, None),
    'k_reference_dm_local_dist': (None, None, 2, None),
    'k_reference_dm_conv': (None, None, None),
    'k_reference_dm_conv_local_dist': (None, None, None),
    'seq_target': (None, PROTEIN_BOW_DIM),
    'seq_refs': (None, PROTEIN_BOW_DIM, None),
}

LOCAL = '/Users/omerronen/Documents/MSA-Completion' in os.getcwd()

PREDICTON_FEATURES = {
    'modeller_dm', 'plmc_score', 'target_pdb_dm', 'reference_dm'
}
DATASET_PATH = '/cs/zbio/orzuk/projects/ContactMaps/src/Periscope/data'

MSA_DATA_PATH = '/cs/zbio/orzuk/projects/ContactMaps/data/MSA_Completion/msa_data'

MSA_STRUCTURES_DATA_PATH = os.path.join(MSA_DATA_PATH, 'structures')

HHBLITS_PATH = os.path.join(MSA_DATA_PATH, 'hhblits')
CCMPRED_PATH = os.path.join(MSA_DATA_PATH, 'ccmpred')
EVFOLD_PATH = '/cs/zbio/orzuk/projects/ContactMaps/data/MSA_Completion/msa_data/dca'
PDB_PATH = '/cs/zbio/orzuk/projects/ContactMaps/data/MSA-Completion/data/pdb'
if LOCAL:
    PDB_PATH = '/Users/omerronen/Documents/MSA-Completion/data/pdb'
    MSA_STRUCTURES_DATA_PATH = PDB_PATH
MODELLER_PATH = '/cs/zbio/orzuk/projects/ContactMaps/data/MSA-Completion/models/Modeller_data'
MODELS_PATH = '/cs/zbio/orzuk/projects/ContactMaps/src/Periscope/models'
if LOCAL:
    MODELS_PATH = '/Users/omerronen/Documents/MSA-Completion/models'

periscope_path = '/cs/zbio/orzuk/projects/ContactMaps/src/Periscope'
if LOCAL:
    periscope_path = '/Users/omerronen/Documents/MSA-Completion'
NUM_HOMOLOGOUS = 500

Architectures = namedtuple('Architectures',
                           'conv references_resnet multi_structure multi_structure_pssm multi_structure_ccmpred'
                           ' multi_structure_ccmpred_2 ms_ccmpred_pssm ms_ss_ccmpred_pssm ms_ss_ccmpred ms_ss_ccmpred_2')

ARCHS = Architectures(conv='conv',
                      references_resnet='references_resnet',
                      multi_structure='multi_structure',
                      multi_structure_pssm='multi_structure_pssm',
                      multi_structure_ccmpred='multi_structure_ccmpred',
                      multi_structure_ccmpred_2='multi_structure_ccmpred_2',
                      ms_ccmpred_pssm='ms_ccmpred_pssm',
                      ms_ss_ccmpred_pssm='ms_ss_ccmpred_pssm',
                      ms_ss_ccmpred='ms_ss_ccmpred',
                      ms_ss_ccmpred_2='ms_ss_ccmpred_2')


def csv_to_list(filename):
    data = pd.read_csv(filename, index_col=0)
    return list(data.iloc[:, 0])


def generate_dataset(is_local):
    if not is_local:
        datasets = {
            'train':
                yaml_load(os.path.join(periscope_path, 'data',
                                       'train_valid.yaml'))['proteins'],
            'train_batch':
                yaml_load(
                    os.path.join(periscope_path, 'data',
                                 'train_valid_batch.yaml')),
            'eval':
                yaml_load(os.path.join(periscope_path, 'data',
                                       'eval_valid.yaml'))['proteins'],
            'pfam':
                yaml_load(os.path.join(periscope_path, 'data',
                                       'pfam_valid.yaml'))['proteins'],
            'testing': ['1d4oA', '4xeaA'],
            'testing_grouped': {
                500: ['1d4oA', '4xeaA'] * 20
            }
        }

    else:
        datasets = {
            'train':
                csv_to_list(
                    '/Users/omerronen/Documents/MSA-Completion/data/training_proteins.csv'
                ),
            'eval':
                csv_to_list(
                    '/Users/omerronen/Documents/MSA-Completion/data/validation_proteins.csv'
                ),
            'pfam':
                csv_to_list(
                    '/Users/omerronen/Documents/MSA-Completion/data/150pfam.csv'),
            'testing': ['1d4oA', '4xeaA']
        }
    return datasets


try:
    DATASETS = generate_dataset(LOCAL)
except FileNotFoundError:
    DATASETS = {}
    ds = {
        'eval': "validation_proteins.csv",
        'train': 'training_proteins.csv',
        'pfam': '150pfam.csv'
    }
    for dataset in ds:
        DATASETS[dataset] = csv_to_list(
            os.path.join(periscope_path, 'data', ds[dataset]))

AMINO_ACID_STATS = {'A': 8.76, 'R': 5.78, 'N': 3.93, "D": 5.49, 'C': 1.38, "Q": 3.9,
                    'E': 6.32, 'G': 7.03, 'H': 2.26, 'I': 5.49, 'L': 9.68, 'K': 5.19,
                    'M': 2.32, 'F': 3.87, 'P': 5.02, 'S': 7.14, 'T': 5.53, 'W': 1.25,
                    'Y': 2.91, 'V': 6.73}
