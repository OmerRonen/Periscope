import os

from .protein import Protein
from .globals import csv_to_list
from .utils_old import yaml_save

LOCAL = os.getcwd() == '/Users/omerronen/Documents/MSA-Completion'

train_proteins = list(
    set(csv_to_list('data/training_proteins.csv')).difference(
        set(csv_to_list('data/bad_proteins.csv'))))

lengths = {}

for protein in train_proteins:
    p, c = protein[0:4], protein[4]
    prot = Protein(protein=p, chain=c)
    l = (len(prot.sequence) // 50) * 50 + 50
    if l in lengths:
        lengths[l].append(protein)
    else:
        lengths[l] = [protein]

yaml_save(filename='data/training_proteins_grouped.yml', data=lengths)
