import os
import subprocess

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from ..data.creator import DataCreator
from ..utils.constants import PATHS
from ..utils.protein import Protein
from ..utils.utils import check_path, run_clustalo


def main():
    protein = '2jp1A'
    family = 'XCL1'

    DataCreator(protein, family=family).ccmpred
    # alt = '2jp1A'
    #
    # org = '2n54A'
    # ans = '7jh1A'
    # prots = [alt, org, ans]
    #
    # sequneces = [SeqRecord(Seq(Protein(p[0:4], p[4]).str_seq), id=p, name=p, description='') for p in prots]
    # path_XCL1 = os.path.join(PATHS.data, 'families', 'XCL1')
    # check_path(path_XCL1)
    # fname = os.path.join(path_XCL1, 'seed.fasta')
    # run_clustalo(sequneces, fname, family="XCL1")
    # target_hhblits_path = path_XCL1
    # target = 'msa'
    # output_hhblits = os.path.join(target_hhblits_path, target + '.a3m')
    # output_reformat1 = os.path.join(target_hhblits_path, target + '.a2m')
    # output_reformat2 = os.path.join(target_hhblits_path, target + '_v%s.fasta' % 2)
    #
    # db_hh = '/vol/sci/bio/data/or.zuk/projects/ContactMaps/data/uniref30/UniRef30_2020_06'
    #
    # hhblits_params = '-n 3 -e 1e-8 -maxfilt 10000000000 -neffmax 20 -nodiff -realign_max 10000000000'
    #
    # hhblits_cmd = f'hhblits -i {fname} -d {db_hh} {hhblits_params} -oa3m {output_hhblits}'
    # subprocess.run(hhblits_cmd, shell=True)
    # # subprocess.run(hhblits_cmd, shell=True, stdout=open(os.devnull, 'wb'))
    # reformat = ['reformat.pl', output_hhblits, output_reformat1]
    # subprocess.run(reformat)
    #
    # reformat = ['reformat.pl', output_reformat1, output_reformat2]
    # subprocess.run(reformat)


if __name__ == '__main__':
    main()
