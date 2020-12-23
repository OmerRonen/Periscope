import subprocess
import os
import tempfile

from Bio import SeqIO, Align
from Bio.Phylo.PAML import codeml
from Bio.AlignIO.PhylipIO import SequentialPhylipWriter

from ..utils.utils import get_aln_fasta, get_target_path, check_path


def _save_phy_aln(fasta_fname, phy_fname, n_seqs=None):
    if os.path.isfile(phy_fname):
        return
    records = SeqIO.parse(fasta_fname, "fasta")
    records_phy = []
    records_phy_names = []
    i = 0
    for record in records:
        if record.id[0:8] not in records_phy_names:
            record.id = record.id[0:8]
            records_phy.append(record)
            records_phy_names.append(record.id[0:8])
            i += 1
            if i == n_seqs:
                break

    aln = Align.MultipleSeqAlignment(records_phy)

    handle = open(phy_fname, 'w')
    pw = SequentialPhylipWriter(handle)
    pw.write_alignment(aln)
    handle.close()


def _infer_tree(phy_fname, tree_fname):
    if os.path.isfile(tree_fname):
        return
    tmp_file = tempfile.NamedTemporaryFile()
    echo_cmd = f"echo -e '\n{tmp_file.name}'"
    fprot_cmd = f'fprotpars {phy_fname} -outtreefile {tree_fname}'

    cmd = f"{echo_cmd} | {fprot_cmd}"
    subprocess.run(cmd, shell=True)


def find_ancestors(protein, n_seqs):
    paml_path = os.path.join(get_target_path(protein), 'paml')
    check_path(paml_path)

    fasta_fname = get_aln_fasta(protein)
    phy_fname = os.path.join(paml_path, 'msa.phy')
    _save_phy_aln(fasta_fname, phy_fname, n_seqs)
    tree_fname = os.path.join(paml_path, 'prot.tree')
    _infer_tree(phy_fname, tree_fname)
    cml_fname = os.path.join(paml_path, 'cml')
    cml = codeml.Codeml()
    cml.read_ctl_file("/vol/sci/bio/data/or.zuk/projects/ContactMaps/src/paml4.9j/codeml.ctl")
    cml.set_options(aaRatefile="/vol/sci/bio/data/or.zuk/projects/ContactMaps/src/paml4.9j/dat/jones.dat")
    cml.tree = tree_fname
    cml.out_file = cml_fname
    cml.working_dir = paml_path
    cml.alignment = phy_fname
    cml.run()

