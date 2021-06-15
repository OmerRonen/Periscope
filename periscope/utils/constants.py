import os
import pathlib
import yaml

import pandas as pd

from collections import namedtuple


def check_path(pth):
    if not os.path.exists(pth):
        pathlib.Path(pth).mkdir(parents=True)


def yaml_load(filename):
    with open(filename, 'r') as stream:
        data_loaded = yaml.load(stream)

    return data_loaded


N_LAYERS_PROJ = 3
N_REFS = 10
C_MAP_PRED = 'c_map_prediction'
UPPER_TRIANGULAR_MSE_LOSS = 'upper_triangular_mse_loss'
UPPER_TRIANGULAR_CE_LOSS = 'upper_triangular_cross_entropy_loss'
CONTACT_MAP_PH = 'contact_map_ph'

Features = namedtuple(
    'Features',
    'target_pdb_cm target_pdb_dm evfold  '
    'reference_dm  k_reference_dm k_reference_dm_conv '
    'seq_target seq_refs seq_target_pssm seq_refs_pssm ccmpred seq_target_pssm_ss seq_refs_pssm_ss seq_target_ss '
    'seq_refs_ss pwm_w pwm_evo conservation beff properties_target')

FEATURES = Features(
    target_pdb_cm='target_pdb_cm',
    target_pdb_dm='target_pdb_dm',
    evfold='evfold',
    reference_dm='reference_dm',
    k_reference_dm='k_reference_dm',
    k_reference_dm_conv='k_reference_dm_conv',
    seq_target='seq_target',
    seq_refs='seq_refs',
    seq_target_pssm='seq_target_pssm',
    seq_refs_pssm='seq_refs_pssm',
    ccmpred='ccmpred',
    seq_target_pssm_ss='seq_target_pssm_ss',
    seq_refs_pssm_ss='seq_refs_pssm_ss',
    seq_refs_ss='seq_refs_ss',
    seq_target_ss='seq_target_ss',
    pwm_w='pwm_w',
    pwm_evo='pwm_evo',
    conservation='conservation',
    beff='beff',
    properties_target='properties_target'
)

PROTEIN_BOW_DIM = 22
PROTEIN_BOW_DIM_SS = 30
PROTEIN_BOW_DIM_PSSM = 42
PROTEIN_BOW_DIM_PSSM_SS = 50
SEQ_ID_THRES = 0.95
FEATURES_DIMS = {
    'target_pdb_cm': (None, None, 1),
    'target_pdb_dm': (None, None, 1),
    'evfold': (None, None, 1),
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

# LOCAL = '/Users/omerronen/Documents/Periscope' in os.getcwd()
# _base_path = '/vol/sci/bio/data/or.zuk/projects/ContactMaps'
# _msa_path = f'{_base_path}/data/MSA_Completion/msa_data'
# _local_pdb_path = '/Users/omerronen/Documents/Periscope/data/pdb'
# _msa_structures_path = os.path.join(_msa_path, 'structures') if not LOCAL else _local_pdb_path
# _local_models_path = '/Users/omerronen/Google Drive (omerronen10@gmail.com)/Periscope/models'
# _models_path = f'{_base_path}/src/Periscope/models' if not LOCAL else _local_models_path
# _local_periscope_path = '/Users/omerronen/Documents/Periscope'
# _periscope_path = f'{_base_path}/src/Periscope' if not LOCAL else _local_periscope_path
# _drive_path = '/Users/omerronen/Google Drive (omerronen10@gmail.com)/Periscope'
# _data_path = os.path.join(_drive_path, 'data') if LOCAL else os.path.join(_periscope_path, 'data')

# A class for all the relevant paths
# drive - google drive path for saving models and artifacts
# data - generic path to save data in
# hhblits - path to store alignment files
# pdb - path for pdb files
# modeller - modeller stuff
# periscope - the curreמt project directory
# proteins - path for post processing proteins data
# src - path to other software we need, i.e dssp, convfold raptor
# raptor - raptor 3d modelling path

local = os.system('hostname') == "Omers-MacBook-Pro-2.local"
local_src = "/Users/omerronen/Documents/Phd/BinProtein"
stanford_src = "/oak/stanford/groups/euan/proteins"
src = local_src if local else stanford_src
prscope_pth = os.path.join(src, "Periscope")
data_pth = os.path.join(prscope_pth, "data")
check_path(data_pth)
hhblits_path = os.path.join(data_pth, "hhblits")
check_path(hhblits_path)
pdb_path = os.path.join(data_pth, "pdb")
check_path(pdb_path)
modeller_path = os.path.join(prscope_pth, "modeller")
check_path(modeller_path)
models_path = os.path.join(prscope_pth, "models")
check_path(models_path)
proteins_path = os.path.join(data_pth, "proteins")
check_path(proteins_path)
raptor_path = f'{prscope_pth}/src/RaptorX-3DModeling/DL4DistancePrediction4/Scripts'
ccmpred_path = os.path.join(src, "CCMpred/bin/ccmpred")
hh_ds = os.path.join(data_pth, "UniRef30_2020_06")


Paths = namedtuple(
    'Paths',
    'drive data hhblits pdb modeller models periscope proteins src raptor ccmpred hh_ds')
PATHS = Paths(
    drive='/Users/omerronen/Google Drive (omerronen10@gmail.com)/Periscope/models',
    data=data_pth,
    hhblits=hhblits_path,
    pdb=pdb_path,
    modeller=modeller_path,
    models=models_path,
    periscope=prscope_pth,
    proteins=proteins_path,
    src=src,
    raptor=raptor_path,
    ccmpred=ccmpred_path,
    hh_ds=hh_ds
)

PREDICTON_FEATURES = {
    'modeller_dm', 'plmc_score', 'target_pdb_dm', 'reference_dm', 'ccmpred'
}

NUM_HOMOLOGOUS = 500

Architectures = namedtuple('Architectures',
                           'conv references_resnet multi_structure multi_structure_pssm multi_structure_ccmpred'
                           ' multi_structure_ccmpred_2 ms_ccmpred_pssm ms_ss_ccmpred_pssm ms_ss_ccmpred ms_ss_ccmpred_2'
                           ' periscope periscope2 templates evo periscope_properties')

ARCHS = Architectures(conv='conv',
                      references_resnet='references_resnet',
                      multi_structure='multi_structure',
                      multi_structure_pssm='multi_structure_pssm',
                      multi_structure_ccmpred='multi_structure_ccmpred',
                      multi_structure_ccmpred_2='multi_structure_ccmpred_2',
                      ms_ccmpred_pssm='ms_ccmpred_pssm',
                      ms_ss_ccmpred_pssm='ms_ss_ccmpred_pssm',
                      ms_ss_ccmpred='ms_ss_ccmpred',
                      ms_ss_ccmpred_2='ms_ss_ccmpred_2',
                      periscope='periscope',
                      periscope2='periscope2',
                      templates='templates',
                      evo='evo',
                      periscope_properties="periscope_properties")

Datasets = namedtuple('Datasets', 'train eval pfam testing cameo membrane cameo41')


def csv_to_list(filename):
    data = pd.read_csv(filename, index_col=0)
    return list(data.iloc[:, 0])


def generate_full_dataset():
    datasets = Datasets(
        train=
        csv_to_list(os.path.join(PATHS.data, 'training_proteins.csv')),
        eval=
        csv_to_list(os.path.join(PATHS.data, 'validation_proteins.csv')),

        pfam=csv_to_list(os.path.join(PATHS.data, '150pfam.csv')),

        cameo=csv_to_list(os.path.join(PATHS.data, 'cameo76.csv')),

        cameo41=csv_to_list(os.path.join(PATHS.data, 'cameo41.csv')),

        membrane=csv_to_list(os.path.join(PATHS.data, 'membrane.csv')),

        testing=['1ej0A', '1hh8A', '1kw4A', '1mk0A', '1tqgA', '1fl0A', '1jo8A']

    )
    return datasets


def generate_dataset():
    datasets = Datasets(
        train=
        yaml_load(os.path.join(PATHS.data,
                               'train.yaml'))['proteins'],
        eval=
        yaml_load(os.path.join(PATHS.data,
                               'eval.yaml'))['proteins'],

        pfam=
        yaml_load(os.path.join(PATHS.data,
                               'pfam.yaml'))['proteins'],

        cameo=yaml_load(os.path.join(PATHS.data,
                                     'cameo.yaml'))['proteins'],

        cameo41=yaml_load(os.path.join(PATHS.data,
                                       'cameo41.yaml'))['proteins'],

        membrane=yaml_load(os.path.join(PATHS.data,
                                        'membrane.yaml'))['proteins'],

        testing=['1ej0A', '1hh8A', '1kw4A', '1mk0A', '1tqgA', '1fl0A', '1jo8A']

    )
    return datasets


# DATASETS = generate_dataset()
ERRS = yaml_load(os.path.join(PATHS.data, 'pp_errs.yaml'))['proteins']
DATASETS_FULL = generate_full_dataset()
DATASETS = generate_dataset()

AMINO_ACID_STATS = {'A': 8.76, 'R': 5.78, 'N': 3.93, "D": 5.49, 'C': 1.38, "Q": 3.9,
                    'E': 6.32, 'G': 7.03, 'H': 2.26, 'I': 5.49, 'L': 9.68, 'K': 5.19,
                    'M': 2.32, 'F': 3.87, 'P': 5.02, 'S': 7.14, 'T': 5.53, 'W': 1.25,
                    'Y': 2.91, 'V': 6.73}
