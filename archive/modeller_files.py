import os
import sys
import logging

logging.getLogger('Modeller_files').setLevel(logging.INFO)

target_protein = sys.argv[1]
target_chain = sys.argv[2]
template_protein = sys.argv[3]
template_chain = sys.argv[4]
version = sys.argv[5]
n_structures = int(sys.argv[6])
src_path = '/cs/zbio/orzuk/projects/ContactMaps/src/Periscope'
modeller_path = '/cs/zbio/orzuk/projects/ContactMaps/data/MSA-Completion/models/Modeller_data'
target = target_protein + target_chain
template = template_protein + template_chain
modeller_files = [
    target + '-' + template + '.ali', target + '.B99990001.pdb',
    target + '.D00000001', target + '.V99990001', target + '.ini',
    target + '.rsr', target + '.sch',
    'pdb' + template_protein + '.ent'
]
for file in modeller_files:
    if not os.path.exists(os.path.join(src_path, file)):
        raise FileNotFoundError('Modeller output file %s not found' % file)
for i in range(n_structures):
    n = str(i + 1)
    predicted_structure = target + f'.B9999000{n}.pdb'
    if n == '10':
        predicted_structure = target + f'.B999900{n}.pdb'
    target_path = os.path.join(modeller_path, target)
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    target_pdb_fname = 'v%s_pdb' % version + target_protein + f'{n}.ent'

    target_modeller_pdb_file = os.path.join(target_path, target_pdb_fname)
    os.rename(os.path.join(src_path, predicted_structure),
              target_modeller_pdb_file)

if os.path.isfile(target_modeller_pdb_file):
    logging.info('saved to %s' % target_modeller_pdb_file)
