from modeller import *
from modeller.automodel import *
import os
from os import path
import sys
import subprocess
import logging

src_path = '/cs/zbio/orzuk/projects/ContactMaps/src/Periscope'
modeller_path = '/cs/zbio/orzuk/projects/ContactMaps/data/MSA-Completion/models/Modeller_data'
pdb_path = ''

LOGGER = logging.getLogger(__name__)


def check_pdb(pdb_fname, pdb_path):
    protein_pdb_file = path.join(pdb_path, pdb_fname)

    if not path.isfile(protein_pdb_file):
        protein = protein_pdb_file[3:7]

        msg = "cannot find a pdb file named '" + protein_pdb_file + "'."
        LOGGER.info(msg)

        pdb_file_zipped = 'pdb%s.ent.gz' % protein

        ftp_file = 'ftp://ftp.wwpdb.org/pub/pdb/data/structures/divided/pdb/%s/%s' % (
            protein[1:3], pdb_file_zipped)

        ftp_cmd = "wget %s -P %s" % (ftp_file, pdb_path + '/')

        subprocess.Popen(ftp_cmd, shell=True).wait()
        cmd2 = 'gunzip -f ' + path.join(pdb_path, pdb_file_zipped)
        subprocess.Popen(cmd2, shell=True).wait()


target = sys.argv[1]
target_chain = sys.argv[2]
target_path = path.join(modeller_path, target + target_chain)
if not path.exists(target_path):
    os.mkdir(target_path)
# target_pdb_fname = 'pdb' + target + '.ent'
target_pir_fname = target+target_chain + '.ali'
# target_pdb_file = path.join(pdb_path, target_pdb_fname)
# check_pdb(target_pdb_fname, pdb_path)
template = sys.argv[3]
template_chain = sys.argv[4]
n_structures = sys.argv[5]
template_pdb_fname = 'pdb' + template + '.ent'
template_pdb_file = path.join(pdb_path, template_pdb_fname)
check_pdb(template_pdb_fname, pdb_path)
alignment_file = '%s-%s.ali' % (target + target_chain,
                                template + template_chain)

env = environ()
aln = alignment(env)
mdl = model(env, file=template, model_segment=('FIRST:%s'%template_chain, 'LAST:%s'%template_chain))
aln.append_model(mdl,
                 align_codes=template + template_chain,
                 atom_files=template_pdb_file)
# mdl = model(env, file=template, model_segment=('FIRST:A', 'LAST:A'))
# aln.append_model(mdl,
#                  align_codes=template + template_chain,
#                  atom_files=template_pdb_file)
aln.append(file=target_pir_fname, align_codes=target.lower()+target_chain)
aln.align2d()
aln.write(file=alignment_file, alignment_format='PIR')


class MyModel(automodel):
    def special_patches(self, aln):
        # Rename both chains and renumber the residues in each
        self.rename_segments(segment_ids=[target_chain], renumber_residues=[1])


env = environ()
env.io.atom_files_directory = ['.', 'atom_files']
a = MyModel(env,
            alnfile=alignment_file,
            knowns=template + template_chain,
            sequence=target + target_chain,
            assess_methods=assess.normalized_dope)
a.starting_model = 1
a.ending_model = int(n_structures)
a.make()
