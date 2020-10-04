# Comparative modeling with multiple templates
from modeller import *  # Load standard Modeller classes
from modeller.automodel import *  # Load the automodel class

import argparse
import logging
import os
import subprocess

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Modeller with multiple templates")

    parser.add_argument('target', type=str)
    parser.add_argument('-a', '--alignment', type=str)
    parser.add_argument('-t', '--templates', nargs='+')
    parser.add_argument('-n', '--structures', type=int)

    return parser.parse_args()


def check_pdb(protein):
    pdb_fname = 'pdb' + protein + '.ent'
    pdb_path = os.getcwd()
    protein_pdb_file = os.path.join(pdb_path, pdb_fname)

    if not os.path.isfile(protein_pdb_file):
        msg = "cannot find a pdb file named '" + protein_pdb_file + "'."
        LOGGER.info(msg)

        pdb_file_zipped = 'pdb%s.ent.gz' % protein

        ftp_file = 'ftp://ftp.wwpdb.org/pub/pdb/data/structures/divided/pdb/%s/%s' % (
            protein[1:3], pdb_file_zipped)

        ftp_cmd = "wget %s -P %s -q" % (ftp_file, pdb_path + '/')

        subprocess.Popen(ftp_cmd, shell=True).wait()
        cmd2 = 'gunzip -f ' + os.path.join(pdb_path, pdb_file_zipped)
        subprocess.Popen(cmd2, shell=True).wait()


class MyModel(automodel):
    def special_patches(self, aln):
        # Rename both chains and renumber the residues in each
        self.rename_segments(segment_ids=[target_chain], renumber_residues=[1])


def main():
    args = parse_args()
    aln_file = args.alignment
    target = args.target
    known = tuple(args.templates)
    check_pdb(target[0:4])
    n_structures = args.structures
    for t in args.templates:
        check_pdb(t[0:4])

    # log.verbose()  # request verbose output
    env = environ()  # create a new MODELLER environment to build this model in

    # directories for input atom files
    env.io.atom_files_directory = ['.', '../atom_files']
    # env.io.hetatm = True
    a = automodel(env,
                  alnfile=aln_file,  # alignment filename
                  knowns=known,  # codes of the templates
                  sequence=target)  # code of the target
    a.starting_model = 1  # index of the first model
    a.ending_model = n_structures  # index of the last model
    # (determines how many models to calculate)
    a.make()


if __name__ == '__main__':
    main()
