import os
import sys
import logging


target_protein = sys.argv[1]
target_chain = sys.argv[2]
version = sys.argv[3]
n_structures = int(sys.argv[4])

files_path = sys.argv[5]
modeller_path = sys.argv[6]
src_path = '/cs/zbio/orzuk/projects/ContactMaps/src/Periscope'
target = target_protein + target_chain


logging.basicConfig(level=logging.INFO)

LOGGER = logging.getLogger(__name__)

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
    os.rename(os.path.join(files_path, predicted_structure),
              target_modeller_pdb_file)

if os.path.isfile(target_modeller_pdb_file):
    LOGGER.info('saved to %s' % target_modeller_pdb_file)

files = os.listdir(files_path)
files_to_remove = [
    os.path.join(files_path, f) for f in files
    if f.endswith('.pdb') or f.endswith('sch') or 'D00000' in f
    or 'V9999' in f or f.endswith('rsr') or f.endswith('ali')
    or f.endswith('ini') or f.endswith('ent')
]
for file in files_to_remove:
    os.remove(file)
