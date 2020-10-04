import os
import csv

import pandas as pd
import numpy as np

data_path = '/Users/omerronen/Documents/MSA-Completion/data'

train_file = '/Users/omerronen/Downloads/cullpdb_pc25_res2.5_R1.0_d200402_chains13467'

pfam_dir = '/Users/omerronen/Downloads/suppdata_1/seq'
pfam_proteins = []
for prot_file in os.listdir(pfam_dir):
    pfam_proteins.append(prot_file.strip('.fasta'))

train_proteins = []
for line in open(train_file, 'r').readlines():
    prot_name = line.split(' ')[0]
    if len(prot_name) == 5:
        protein = prot_name[0:4].lower()
        chain = prot_name[4]
        train_proteins.append(protein + chain)

train_proteins_set = pd.DataFrame(set(train_proteins))
train_set, validation_set = np.split(train_proteins_set.sample(frac=1),
                                     [int(.8 * len(train_proteins_set))])

train_set = list({p[0]
                  for p in train_set.values.tolist()
                  }.difference(pfam_proteins))
validation_set = list({p[0]
                       for p in validation_set.values.tolist()
                       }.difference(pfam_proteins))


def save_to_csv(protein_list, dataset_name):
    pfam_df = pd.DataFrame(protein_list, columns=['protein'])
    pfam_df.to_csv(os.path.join(data_path, '%s.csv' % dataset_name))


save_to_csv(pfam_proteins, '150pfam')
save_to_csv(train_set, 'training_proteins')
save_to_csv(validation_set, 'validation_proteins')

print(set(pfam_proteins).intersection(set(train_proteins)))
