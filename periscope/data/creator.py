import collections
import itertools
import logging
import os
import re
import time
import subprocess
import tempfile
import urllib

import numpy as np
import pandas as pd

from scipy.special import softmax
from Bio import SeqIO, pairwise2
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from .aligner import Aligner
from .property import get_properties
from .pssm import compute_PWM
from ..utils.protein import Protein
from ..utils.constants import PATHS, DATASETS, PROTEIN_BOW_DIM, SEQ_ID_THRES, N_REFS
from ..utils.utils import (convert_to_aln, write_fasta, MODELLER_VERSION, create_sifts_mapping, read_raw_ec_file,
                           pkl_save, pkl_load, compute_structures_identity_matrix, VERSION, get_modeller_pdb_file,
                           get_target_path, get_target_ccmpred_file, check_path, read_fasta, run_clustalo,
                           get_aln_fasta, get_predicted_pdb, save_chain_pdb, get_a3m_fname, get_target_scores_file,
                           get_target_hhblits_path, get_fasta_fname)

logging.basicConfig(level=logging.INFO)

LOGGER = logging.getLogger(__name__)

SIFTS_MAPPING = create_sifts_mapping()


class DataCreator:
    _VERSION = VERSION
    _MSA_FULL_VERSION = 3
    _MSA_VERSION = 2
    _EVFOLD_VERSION = 2
    _PSSM_VERSION = 2
    NAN_VALUE = -1.0
    SEQUENCE_LENGTH_THRESHOLD = 50
    _MODELLER_VERSION = MODELLER_VERSION
    _STRUCTURES_VERSION = 3
    _PHYLO_VERSION = 7
    _THRESHOLD = 8

    def __init__(self, target, n_refs=N_REFS, family=None, require_template=True, train=True):
        s = time.time()
        self.protein = Protein(target[0:4], target[4])
        self.str_seq = self.protein.str_seq
        self._train = train
        try:
            self._is_broken = not hasattr(self.protein, 'sequence')
        except Exception:
            self._is_broken = True
        self._family = family
        self._msa_data_path = os.path.join(get_target_path(target, family), 'features')
        check_path(self._msa_data_path)
        self.target = target
        self.has_msa = os.path.isfile(get_aln_fasta(self.target, self._family))
        if not self.has_msa and train:
            self.has_refs = False
            return
        self.aligner = Aligner(self.target, self._family)
        if self._family is not None:
            LOGGER.info(f'Family {self._family}')
            self.target_seq_msa = np.array(list(self._parse_msa()[self.target]))
            seq = self.str_seq
            self.str_seq = "".join(self.target_seq_msa[self.target_seq_msa != '-']) if seq is None else seq
        self._n_refs = n_refs
        self.metadata = self._get_metadata()

        self.has_refs = self.aligner.has_templates if self.has_msa else False
        self.refactored = self.metadata['refactored']
        self.recreated = self.metadata.get('new_data', False)
        if not os.path.isfile(self.fasta_fname):
            self._write_fasta()
        self._require_template = require_template
        LOGGER.info(f'init took {time.time() - s}')

    def generate_data(self):
        self.ccmpred
        self.raptor_properties

        self.aligner.templates_ss_acc_seq_tensor
        self.aligner.templates_distance_tensor
        self.pwm_w

    @property
    def msa_length(self):
        return len(self._parse_msa())

    @property
    def sorted_structures(self):
        structures_sorted_file = os.path.join(
            self._msa_data_path, 'structures_sorted_tst.pkl')

        sorted_structures = pkl_load(structures_sorted_file)

        if sorted_structures is not None:
            return sorted_structures

        struc_dm = self._get_phylo_structures_mat()
        if struc_dm is None:
            return
        struc_dm = struc_dm.loc[self.target, :]
        struc_dm_valid = struc_dm[struc_dm > 1 - SEQ_ID_THRES].index

        sorted_structures = struc_dm[struc_dm_valid].sort_values(ascending=False)

        if sorted_structures is None:
            return

        if len(sorted_structures) == 0:
            return

        pkl_save(filename=structures_sorted_file, data=sorted_structures)
        return sorted_structures

    def _get_required_files(self):
        pssm_file = f'pssm_bio_v{self._MSA_VERSION}.pkl'
        seq_target = 'seq_target.pkl'
        metadata_file = 'meta_data.pkl'
        target_seq_file = 'target_seq.pkl'
        refs_seqs_file = 'refs_seqs.pkl'

        return [pssm_file, seq_target, metadata_file, target_seq_file, refs_seqs_file]

    def _clean_target_folder(self):
        files = os.listdir(self._msa_data_path)

        def _is_struc_file(f):
            return f.endswith(f'_mean_v3.pkl') or f.endswith(f'_structure_v3.pkl')

        for f in files:
            if f not in self._get_required_files() and not _is_struc_file(f):
                os.remove(os.path.join(self._msa_data_path, f))

    def get_no_target_gaps_msa(self, sub=False):

        msa_file = get_aln_fasta(self.target, self._family)

        if not os.path.isfile(msa_file):
            self._run_hhblits()

        fasta_seqs = self._parse_msa()
        if self._family is not None and not sub:
            return list(fasta_seqs.values())
        target_seq_full = fasta_seqs[self.target]
        target_seq_no_gap_inds = [i for i in range(len(target_seq_full)) if target_seq_full[i] != '-']
        target_seq = ''.join(target_seq_full[i] for i in target_seq_no_gap_inds)

        def _slice_seq(seq, inds):
            seq.seq = Seq(''.join(seq.seq[i] for i in inds))
            return seq

        assert target_seq == self.protein.str_seq if self._family is None else self.str_seq

        fasta_seqs_short = [_slice_seq(s, target_seq_no_gap_inds) for s in fasta_seqs.values() if s.id != self.target]
        fasta_seqs_short = [_slice_seq(fasta_seqs[self.target], target_seq_no_gap_inds)]+fasta_seqs_short
        return fasta_seqs_short

    def _get_cover(self, ref):
        return self.metadata['references_map'][ref][0][0]

    def get_plot_reference_data(self, ref):

        parsed_msa = self._parse_msa()

        ref_pdb = self._get_cover(ref)

        full_ref_msa = np.array(parsed_msa[ref].seq)
        full_target_msa = np.array(parsed_msa[self.target].seq)

        seq_ref_msa = ''.join(full_ref_msa[full_target_msa != '-'])
        seq_target_msa = ''.join(full_target_msa[full_target_msa != '-'])

        ref_title = f'{ref} ({ref_pdb})'
        target_title = f'Target ({self.target})'
        if len(ref_title) < len(target_title):
            diff = len(target_title) - len(ref_title)
            ref_title += diff * ' '
        elif len(ref_title) > len(target_title):
            diff = len(ref_title) - len(target_title)
            target_title += diff * ' '

        msg = f'{target_title} : {seq_target_msa}\n{ref_title} : {seq_ref_msa}'
        phylo_dm = self._get_phylo_structures_mat()
        seq_dist = np.round(phylo_dm.loc[ref, self.target], 2)
        data = {'msg': msg, 'sequence_distance': float(seq_dist)}

        return data

    def _get_modeller_pir(self):

        # seqs = []
        #
        # msa = self._parse_msa()
        structures = list(self.sorted_structures.index)
        structures = structures if len(structures) <= self._n_refs else structures[0:self._n_refs]
        ref_map = self.aligner.get_ref_map()
        templates_list = [ref_map[s] for s in structures]
        LOGGER.info(structures)

        def _get_description(target):
            protein = target[0:4]
            chain = target[4]

            start, end = Protein(protein, chain).modeller_start_end

            description = f'structure:{protein}:{start}:{chain}:{end}:{chain}::::'

            return description

        def _get_record(t):

            prot = Protein(t[0:4], t[4])
            seq = prot.str_seq

            return SeqRecord(Seq(seq), id=t, name='', description=_get_description(t))

        aln_list = [self.target] + templates_list
        aln_list = [t for t in aln_list if hasattr(Protein(t[0:4], t[4]), 'str_seq')]

        sequences = [_get_record(t) for t in aln_list]

        with tempfile.NamedTemporaryFile(suffix='.pir') as f:
            run_clustalo(sequences=sequences, fname=f.name)

            aln = read_fasta(f.name, full=True)
            SeqIO.write(aln, f.name, 'pir')
            txt = open(f.name, 'r').readlines()

        def _correct_line(line):
            if ">XX" in line:
                line = line.replace('>XX', '>P1')
            if 'structure' in line:
                line = line.split(" ")[-1]
            return line

        aln_txt = "".join([_correct_line(l) for l in txt])

        # refrences_map = self._metadata['references_map']
        # sequences = {s: str(msa[s].seq) for s in self.known_structures}

        # def _get_hom_txt(hom, seq):
        #     protein = hom[0:4]
        #     chain = hom[4]
        #     line1 = f'>P1;{hom}'
        #
        #     line2 = f'structure:{protein}::{chain}::{chain}::::'
        #
        #     # line2 = f'structure:{protein}::::::::'
        #
        #     line3 = seq + '*\n\n'
        #     return "\n".join([line1, line2, line3])
        #
        # def _create_pir_alignment(refrences_map, sequences, target, structures):
        #     pir_txt = ""
        #
        #     pir_txt += _get_hom_txt(target, sequences[target])
        #     for hom in structures:
        #         template = refrences_map[hom][0][0]
        #
        #         pir_txt += _get_hom_txt(template, sequences[hom])
        #
        #     return pir_txt
        #
        # aln_txt = _create_pir_alignment(refrences_map, sequences, self.target, structures)
        return aln_txt

    def run_modeller_templates(self, n_structures=1, model=None):
        if self.sorted_structures is None:
            return
        sp = model is not None
        file = get_modeller_pdb_file(self.target, n_struc=n_structures, sp=sp)
        if os.path.isfile(file):
            return

        aln_txt = self._get_modeller_pir()
        structures = list(self.sorted_structures.index)
        ref_map = self.aligner.get_ref_map()
        templates_list = [ref_map[s] for s in structures]
        templates_list = templates_list if len(templates_list) <= self._n_refs else templates_list[0:self._n_refs]
        templates = " ".join(templates_list)
        version = self._MODELLER_VERSION

        with tempfile.NamedTemporaryFile(suffix='.pir') as f:
            # SeqIO.write(seqs, f.name, 'pir')
            with open(f.name, 'w') as e:
                e.write(aln_txt)

            args = f'{self.target} -a {f.name} -t {templates} -n {n_structures}'
            outpath = os.path.join(PATHS.modeller, "templates")

            if model is not None:
                LOGGER.info(f'Using {model.name}')
                starting_point = get_predicted_pdb(model, self.target)
                pred_pdb = os.path.join(PATHS.modeller, f'{self.target}_pred.pdb')
                save_chain_pdb(self.target, pred_pdb, starting_point, self.protein.modeller_start_end[0])
                args += f' -s {pred_pdb}'
                outpath = os.path.join(PATHS.modeller, "templates", 'sp')
                LOGGER.info(f'{os.path.isfile(pred_pdb)}')
                if not os.path.isfile(pred_pdb):
                    return
            check_path(outpath)

            modeller_python = '/cs/staff/dina/modeller9.18/bin/modpy.sh python3'
            cmd = f'{modeller_python} -m periscope.data.modeller.run_modeller_templates {args}'
            subprocess.run(cmd, shell=True)
            if model is not None:
                os.remove(pred_pdb)
            mod_args_2 = f'{self.protein.protein} {self.protein.chain}' \
                         f' {version} {n_structures}  {os.getcwd()} {outpath}'

            cmd2 = f'{modeller_python} -m periscope.data.modeller.modeller_files {mod_args_2}'
            subprocess.run(cmd2, shell=True)

    def _run_modeller(self, n_structures):
        """Runs Modeller to compute pdb file from template which is the closest know structure


        """
        if self.closest_reference is None:
            return

        version = self._MODELLER_VERSION
        ref_map = self.aligner.get_ref_map()
        reference = ref_map[self.closest_reference]

        if reference is None:
            LOGGER.info('Reference not found for %s' % self.target)
            return

        template_protein, template_chain = reference[0:4], reference[4]

        mod_args = f'{self.protein.protein} {self.protein.chain} {template_protein} {template_chain} {n_structures}'

        # saving target in pir format
        name = 'sequence:%s:::::::0.00: 0.00' % self.target
        target_seq = SeqIO.SeqRecord(Seq(self.protein.str_seq),
                                     name=name, id=self.target, )
        # SeqIO.PirIO.PirWriter(open(os.path.join(periscope_path, '%s.ali'%self.target), 'w'))
        ali_file = os.path.join(os.getcwd(), '%s.ali' % self.target)
        SeqIO.write(target_seq, ali_file, format='pir')
        with open(ali_file) as f:
            lines = f.readlines()

        # lines  # ['This is the first line.\n', 'This is the second line.\n']

        lines[0] = lines[0].replace("XX", "P1")

        # lines  # ["This is the line that's replaced.\n", 'This is the second line.\n']

        with open(ali_file, "w") as f:
            f.writelines(lines)
        modeller_python = '/cs/staff/dina/modeller9.18/bin/modpy.sh python3'
        cmd = f'{modeller_python} -m periscope.data.modeller.run_modeller {mod_args}'
        try:
            subprocess.run(cmd, shell=True)
        except urllib.error.URLError:
            raise FileNotFoundError('pdb download error')

        mod_args_2 = f'{self.protein.protein} {self.protein.chain}' \
                     f' {version} {n_structures}  {os.getcwd()} {PATHS.modeller}'

        cmd2 = f'{modeller_python} -m periscope.data.modeller.modeller_files {mod_args_2}'
        subprocess.run(cmd2, shell=True)

    def _run_ccmpred(self):

        msa = self.get_no_target_gaps_msa()
        if self._family == 'trypsin':
            l = len(self.target_seq_msa)
            msa = [m for m in msa if len(m) == l]
        if len(msa) > 32000:
            msa = msa[0:32000]

        with tempfile.NamedTemporaryFile() as tmp:
            with tempfile.NamedTemporaryFile(suffix='.aln') as tmp2:
                tmp_name = tmp.name
                tmp2_name = tmp2.name
                write_fasta(msa, tmp_name)

                convert_to_aln(tmp_name, tmp2_name)

                ccmpred_mat_file = get_target_ccmpred_file(self.target, self._family)

                ccmpred_cmd = f'ccmpred {tmp2_name} {ccmpred_mat_file}'
                p = subprocess.run(ccmpred_cmd, shell=True)
                if p.returncode != 0:
                    LOGGER.info(f'CCMpred failed for {self.target}')
                return p.returncode == 0

    def _run_evfold(self):
        # Runs 'plmc' to compute evolutionary couplings

        version = self._EVFOLD_VERSION

        target_version = (self.target, version)
        evfold_path = os.path.join(get_target_path(self.target), 'evfold')
        file_ec = os.path.join(evfold_path, '%s_v%s.txt' % target_version)
        file_params = os.path.join(evfold_path, '%s_%s.params' % target_version)
        hhblits_path = get_target_hhblits_path(self.target)
        file_msa = os.path.join(hhblits_path, '%s_v%s.fasta' % target_version)

        if not os.path.isfile(file_msa):
            self._run_hhblits()

        cmd = f'plmc -o {file_params} -c {file_ec} -le 16.0 -m {500} -lh 0.01  -g -f {self.target} {file_msa}'

        subprocess.call(cmd, shell=True)

    @property
    def fasta_fname(self):
        return get_fasta_fname(self.target, self._family)

    def _write_fasta(self):
        s = self.protein.str_seq if self._family is None else self.str_seq
        if s is None:
            return
        seq = Seq(s.upper())
        sequence = SeqIO.SeqRecord(seq, name=self.target, id=self.target)
        SeqIO.write(sequence, self.fasta_fname, "fasta")

    def _run_hhblits(self):
        # Generates multiple sequence alignment using hhblits
        if self.protein.str_seq is None or self._train:
            return
        seq = Seq(self.protein.str_seq.upper())

        target = self.target
        version = self._MSA_VERSION

        sequence = SeqIO.SeqRecord(seq, name=target, id=target)
        target_hhblits_path = get_target_hhblits_path(target)
        check_path(target_hhblits_path)

        query = os.path.join(target_hhblits_path, target + '.fasta')
        SeqIO.write(sequence, query, "fasta")
        output_hhblits = os.path.join(target_hhblits_path, target + '.a3m')
        output_reformat1 = os.path.join(target_hhblits_path, target + '.a2m')
        output_reformat2 = os.path.join(target_hhblits_path, target + '_v%s.fasta' % version)

        db_hh = PATHS.hh_ds

        hhblits_params = '-n 3 -e 1e-3 -maxfilt 10000000000 -neffmax 20 -nodiff -realign_max 10000000000'

        hhblits_cmd = f'hhblits -i {query} -d {db_hh} {hhblits_params} -oa3m {output_hhblits}'
        subprocess.run(hhblits_cmd, shell=True)
        # subprocess.run(hhblits_cmd, shell=True, stdout=open(os.devnull, 'wb'))
        reformat_script = os.path.join(PATHS.periscope, 'scripts', 'reformat.pl')
        reformat = f"perl {reformat_script} {output_hhblits} {output_reformat1}"
        subprocess.run(reformat, shell=True)

        reformat = f"perl {reformat_script} {output_reformat1} {output_reformat2}"
        subprocess.run(reformat, shell=True)

    def run_hhblits_cns(self):
        # Generates multiple sequence alignment using hhblits

        seq = Seq(self.protein.str_seq_full.upper())
        target = self.target
        version = self._MSA_FULL_VERSION

        sequence = SeqIO.SeqRecord(seq, name=target, id=target)
        query = os.path.join(PATHS.msa, 'query', target + '_full.fasta')
        SeqIO.write(sequence, query, "fasta")
        output_hhblits = os.path.join(PATHS.hhblits, 'a3m', target + '_full.a3m')
        output_reformat1 = os.path.join(PATHS.hhblits, 'a2m', target + '_full.a2m')
        output_reformat2 = os.path.join(PATHS.hhblits, 'fasta',
                                        target + '_full_v%s.fasta' % version)

        db_hh = '/cs/zbio/orzuk/projects/ContactMaps/data/MSA_Completion/hh/uniprot20_2016_02/uniprot20_2016_02'

        hhblits = [
            'hhblits', '-i', query, '-d', db_hh, '-n', '3', '-e', '1e-3',
            '-maxfilt', '10000000000', '-neffmax', '20', '-nodiff', '-realign',
            '-realign_max', '10000000000', '-oa3m', output_hhblits
        ]
        subprocess.run(hhblits)
        reformat = ['reformat.pl', output_hhblits, output_reformat1]
        subprocess.run(reformat)

        reformat = ['reformat.pl', output_reformat1, output_reformat2]
        subprocess.run(reformat)

    def _parse_custom_msa(self):

        msa_file = get_aln_fasta(self.target, self._family)

        def _get_id(seq):
            if "UniRef100" in seq.name:
                return seq.name.split('_')[1], ""
            if len(seq.name) == 5:
                return seq.name, seq.name
            des = seq.description.split('|')
            uniprot_id = des[1]

            if len(des) == 3:
                pdb = ""
                return uniprot_id, pdb
            pdbs = des[3].split("+")
            if self.target in pdbs:
                return None, None
            pdb = pdbs[0].split('_')[0].lower() + pdbs[0].split('_')[1]

            return uniprot_id, pdb

        fasta_seqs = list(SeqIO.parse(msa_file, "fasta"))
        sequences = {}
        for seq in fasta_seqs:
            uniprot_id, pdb_id = _get_id(seq)

            if uniprot_id in sequences or uniprot_id is None:
                continue
            is_target = pdb_id == self.target
            id = pdb_id if is_target else uniprot_id
            # seq.seq = Seq(seq.seq.replace(string.lowercase, '-'))
            # print(len(seq.seq.upper()))
            # seq_arr = np.array(list(seq.seq))
            # seq_arr_upper = np.array(list(seq.seq.upper()))
            # seq_arr[seq_arr!=seq_arr_upper] = "-"
            # print(len(seq_arr))
            is_pdb = len(id) == 5
            seq = Seq(str(seq.seq).upper()) if is_pdb else Seq(re.sub('[a-z]', '-', str(seq.seq)))
            s = SeqRecord(seq, id=id)
            # s = SeqRecord(Seq(re.sub('[a-z]', '-', str(seq.seq))), id=id)  # SeqRecord(seq.seq, id=id)
            sequences[id] = s

        return sequences

    def _parse_msa_default(self):
        """Parses the msa data

        Returns:
            dict[str,SeqRecord]: homologos name to sequence object mapping

        """

        version = self._MSA_VERSION

        parsed_msa_file = os.path.join(self._msa_data_path,
                                       'parsed_msa_v%s.pkl' % version)

        if os.path.isfile(parsed_msa_file):
            return pkl_load(parsed_msa_file)

        msa_file = get_aln_fasta(self.target)

        if not os.path.isfile(msa_file):
            self._run_hhblits()
        if not os.path.isfile(msa_file):
            return

        def _get_id(seq):
            if seq.id == self.target:
                return seq.id
            if "|" in seq.id:
                return seq.id.split('|')[1]
            return seq.id.split('_')[1]

        # fasta_seqs = list(SeqIO.parse(msa_file, "fasta", alphabet=Gapped(ExtendedIUPACProtein())))
        fasta_seqs = list(SeqIO.parse(msa_file, "fasta"))
        sequences = {
            _get_id(seq): SeqRecord(Seq(re.sub('[a-z]', '-', str(seq.seq))),  # seq.seq.upper()
                                    id=_get_id(seq).split("_")[0])
            for seq in fasta_seqs
        }

        return sequences

    def _parse_msa(self):
        if self._family is not None:
            return self._parse_custom_msa()
        return self._parse_msa_default()

    def _get_metadata(self):
        metadata_fname = os.path.join(self._msa_data_path, 'meta_data.pkl')
        has_metadata = os.path.isfile(metadata_fname)
        if has_metadata:
            metadata = pkl_load(metadata_fname)
            metadata['refactored'] = len(metadata.get('references_map', {})) > 0
            return metadata
        metadata = {'references_map': {}, 'refactored': False}
        return metadata

    def _save_metadata(self):
        metadata_fname = os.path.join(self._msa_data_path, 'meta_data.pkl')

        pkl_save(metadata_fname, self.metadata)

    def _get_sorted_structures(self):

        structures_sorted_file = os.path.join(
            self._msa_data_path, 'structures_sorted.pkl')

        struc_dm = self._get_phylo_structures_mat()

        if self.target not in struc_dm or len(self.known_structures) < 2:
            return

        sorted_structures_by_distance = np.clip(struc_dm.loc[self.target, :],
                                                a_min=0.0001,
                                                a_max=1).sort_values(kind='mergesort')

        sorted_structures_by_distance = softmax(
            np.log(1 / sorted_structures_by_distance.loc[
                sorted_structures_by_distance.index != self.target]))

        if len(sorted_structures_by_distance) == 0:
            return

        pkl_save(structures_sorted_file, sorted_structures_by_distance)

        return sorted_structures_by_distance

    def _get_k_closest_references(self):

        sorted_structures_by_distance = self.sorted_structures

        if sorted_structures_by_distance is None:
            return

        structures = sorted_structures_by_distance.sort_values(ascending=True)

        return list(structures.index)

    def _get_seq_target(self):

        seq_target_file = os.path.join(self._msa_data_path, 'target_seq.pkl')

        if os.path.isfile(seq_target_file):
            loaded_seq = np.squeeze(pkl_load(seq_target_file))
            if loaded_seq.shape[0] == self.seq_len:
                return loaded_seq

        bow_msa_target = np.squeeze(self._bow_msa(refs=[self.target]))

        pkl_save(filename=seq_target_file, data=bow_msa_target)

        return bow_msa_target

    @staticmethod
    def _one_hot_msa(numeric_msa):

        msa = np.expand_dims(numeric_msa, 2)

        def _bow_prot(a, axis):
            msa_bow = []

            for hom in a:
                bow = []

                for aa in list(hom):
                    aa_numeric = np.zeros(PROTEIN_BOW_DIM)
                    aa_numeric[int(aa)] = 1.0
                    bow.append(aa_numeric)
                bow_arr = np.array(bow)
                msa_bow.append(bow_arr)
            return np.array(msa_bow)

        bow_msa = np.apply_over_axes(func=_bow_prot, a=msa, axes=[0])

        return bow_msa

    def _save_scores(self):
        a3m_file = get_a3m_fname(self.target, self._family)
        if not os.path.isfile(a3m_file):
            if self._family is None:
                self._run_hhblits()
            else:
                fasta_file = get_aln_fasta(self.target, self._family)
                subprocess.call(f'reformat.pl {fasta_file} {a3m_file}', shell=True)

        fasta_file = get_aln_fasta(self.target, self._family)
        fasta_seqs = list(SeqIO.parse(fasta_file, "fasta"))
        if self._family is not None or len(fasta_seqs) > 10000:

            sub_msa = fasta_seqs[0:10000]
            # sub_msa = list(np.random.choice(fasta_seqs, 10000,p=weights(fasta_seqs)))
            msa_full = self._parse_msa()
            sub_msa = [msa_full[self.target]] + sub_msa
            with tempfile.NamedTemporaryFile(suffix='.fasta') as fasta_tmp:
                write_fasta(sub_msa, fasta_tmp.name)
                with tempfile.NamedTemporaryFile(suffix=".a3m") as a3m_tmp:
                    subprocess.run(f'reformat.pl {fasta_tmp.name} {a3m_tmp.name}', shell=True)
                    scores = compute_PWM(a3m_tmp.name)
        else:
            scores = compute_PWM(a3m_file)
        scores_file = get_target_scores_file(self.target, self._family)
        pkl_save(data=scores, filename=scores_file)
        cmd = f"chgrp prscope {a3m_file}"
        subprocess.call(cmd, shell=True)

    @property
    def scores(self):
        scores_file = get_target_scores_file(self.target, self._family)

        if not os.path.isfile(scores_file):
            if self._train:
                return
            self._save_scores()

        scores = pkl_load(scores_file)

        if scores['conservation'].shape[0] != self.seq_len:
            self._save_scores()

        return pkl_load(scores_file)

    def _fix_scores(self, score):
        if self._family is None:
            return score
        inds = np.where(self.target_seq_msa != '-')
        target_msa_seq_no_gaps = ''.join(self.target_seq_msa[inds])
        sub_ind = self.str_seq.find(target_msa_seq_no_gaps)
        rng = list(range(sub_ind, sub_ind + len(target_msa_seq_no_gaps)))
        l = self.seq_len

        shp = [l] + list(score.shape[1:])

        score_mat = np.zeros(shape=shp)
        idx = np.array(rng)
        score_mat[idx, ...] = score
        return score_mat

    @property
    def pwm_w(self):
        return self._fix_scores(self.scores['PWM'][0])

    @property
    def pwm_evo(self):
        return self._replace_nas(self._fix_scores(self.scores['PWM'][1]))

    @property
    def pwm_evo_ss(self):
        # acc_ss = np.mean(self._get_reference_ss_acc(), axis=2)
        acc_ss_raw = self.aligner.templates_ss_acc_tensor
        if acc_ss_raw is None:
            if self._require_template:
                return
            acc_ss_raw = np.zeros((self.seq_len, 9, self._n_refs))
        acc_ss_raw = self.trim_pad_arr(self._fix_scores(acc_ss_raw))

        acc_ss = np.nanmean(acc_ss_raw, axis=2)
        return self._replace_nas(np.concatenate([self.pwm_evo, acc_ss], axis=-1))

    @property
    def conservation(self):
        return self._fix_scores(np.expand_dims(self.scores['conservation'], axis=1))

    @property
    def beff(self):
        return np.array([self.scores['Beff_alignment']], dtype=np.float32)

    def _bow_msa(self, refs=None, msa=None):
        """

        Args:
            n (int): number of homologous
            refs (list[str]): references to use as sub-alignment

        Returns:
            np.array: of shape (n, l, 22)

        """
        parsed_msa = self._parse_msa() if msa is None else msa
        target_seq = parsed_msa[self.target].seq

        if parsed_msa is None:
            return

        if refs is not None:
            parsed_msa = {r: parsed_msa[r] for r in refs}

        numeric_msa = self._get_numeric_msa(parsed_msa, target_seq)

        return self._one_hot_msa(numeric_msa)

    def _get_numeric_msa(self, parsed_msa, target_seq):

        target_sequence_msa = ''.join(target_seq)
        pdb_sequence = "".join(self.protein.sequence)

        vals = parsed_msa.values()

        homologos = pd.DataFrame(vals).values
        alignment_pdb = np.zeros(shape=(len(homologos), len(pdb_sequence)),
                                 dtype='<U1')
        alignment_pdb[:] = '-'

        pdb_inds, msa_inds = self._align_pdb_msa(
            pdb_sequence, target_sequence_msa, list(range(len(pdb_sequence))))

        alignment_pdb[:, pdb_inds] = homologos[:, msa_inds]

        aa = list('-ACDEFGHIKLMNPQRSTVWYX')
        aa_dict = {aa[i]: i for i in range(len(aa))}
        aa_dict['Z'] = 0
        aa_dict['B'] = 0
        aa_dict['U'] = 0
        aa_dict['O'] = 0

        numeric_msa = np.vectorize(aa_dict.__getitem__)(alignment_pdb).astype(
            np.int32)

        return numeric_msa

    def _generate_seq_refs_full_test(self):

        refs = self._get_k_closest_references()

        if refs is None:
            return

        self._get_clustalo_msa()

        msa = {s.id: s for s in self._aln}

        bow_msa_refs = self._bow_msa(msa=msa).transpose(1, 2, 0)
        bow_msa_refs = bow_msa_refs[..., 1:]
        return bow_msa_refs

    def _generate_seq_refs_full(self):

        seq_refs_file = os.path.join(self._msa_data_path, 'refs_seqs.pkl')

        refs = self._get_k_closest_references()

        if refs is None:
            return

        bow_msa_refs = self._bow_msa(refs=refs).transpose(1, 2, 0)

        pkl_save(filename=seq_refs_file, data=bow_msa_refs)

        return bow_msa_refs

    def _get_seq_refs_test(self):
        """Numeric sequence representation of references

        Returns:
            np.array: of shape (l, PROTEIN_BOW_DIM, k)

        """

        file = os.path.join(self._msa_data_path, 'seq_refs_test_fix.pkl')
        if os.path.isfile(file):
            output = pkl_load(file)
            if output.shape[2] == self._n_refs:
                return output

        bow_msa_full = self._generate_seq_refs_full_test()

        if bow_msa_full is None:
            return
        bow_msa_full = bow_msa_full
        shape = bow_msa_full.shape
        total_refs = shape[-1]

        if total_refs >= self._n_refs:
            output = bow_msa_full[..., 0:self._n_refs]
        else:
            output = np.concatenate(
                [np.zeros((shape[0], shape[1], self._n_refs - total_refs)), bow_msa_full],
                axis=2)

        pkl_save(file, output)
        return output

    @property
    def closest_reference(self):

        sorted_structures = self.sorted_structures
        if sorted_structures is None:
            return
        structure = sorted_structures.index[0]
        assert structure != self.protein.target
        return structure

    @staticmethod
    def _align_pdb_msa(pdb_sequence, msa_sequence, pdb_indices, one_d=False):
        """Get aligned indices if exists

        Args:
            pdb_sequence (str): Amino acid sequence
            msa_sequence (str): Amino acid and gaps sequence
            pdb_indices (list[int]): Pdb indices according to sifts mapping
            one_d (bool): if true we return 1 dimentional

        Returns:
            tuple[list[int, int], list[int, int]]: matching x-y indices for pdb and msa arrays

        """

        msa_sequence_no_gaps = msa_sequence.replace('-', "")
        pdb_msa_substring = msa_sequence_no_gaps.find(pdb_sequence)
        msa_pdb_substring = pdb_sequence.find(msa_sequence_no_gaps)

        msa_indices = [
            i for i in range(len(msa_sequence)) if msa_sequence[i] != '-'
        ]

        if pdb_msa_substring == -1 and msa_pdb_substring == -1:

            return None, None

        elif pdb_msa_substring != -1:

            start_msa = pdb_msa_substring
            end_msa = start_msa + len(pdb_sequence)
            msa_indices = msa_indices[start_msa:end_msa]

        elif msa_pdb_substring != -1:
            start_pdb = msa_pdb_substring
            end_pdb = msa_pdb_substring + len(msa_sequence_no_gaps)
            pdb_indices = pdb_indices[start_pdb:end_pdb]

        if one_d:
            return pdb_indices, msa_indices

        pairs_pdb = list(itertools.combinations(pdb_indices, 2))
        pairs_msa = list(itertools.combinations(msa_indices, 2))

        pdb_inds, msa_inds = list(zip(*pairs_pdb)), list(zip(*pairs_msa))

        if len(pdb_inds) == 0:
            return None, None

        return pdb_inds, msa_inds

    @property
    def known_structures(self):
        return [self.target] + self.aligner.known_structures

    def _get_phylo_structures_mat(self):

        version = self._PHYLO_VERSION

        phylo_structures_file = os.path.join(
            self._msa_data_path, 'structures_phylo_dist_mat_v%s.pkl' % version)

        if os.path.isfile(phylo_structures_file):
            return pkl_load(phylo_structures_file)

        return self._generate_phylo_structures_mat()

    def _generate_phylo_structures_mat(self):

        version = self._PHYLO_VERSION

        phylo_structures_file = os.path.join(
            self._msa_data_path, 'structures_phylo_dist_mat_v%s.pkl' % version)

        parsed_msa = self._parse_msa()

        structures_list = self.known_structures

        if len(structures_list) == 0:
            return
        try:
            msa_structures = [
                str(parsed_msa[s].seq).upper() for s in structures_list
            ]
        except KeyError:
            return
        id_mat = compute_structures_identity_matrix(msa_structures,
                                                    msa_structures,
                                                    target=str(parsed_msa[self.target].seq.upper()))
        identity_mat = pd.DataFrame(id_mat,
                                    columns=structures_list,
                                    index=structures_list)
        phylo_structures_mat = 1 - identity_mat

        phylo_structures_mat.to_pickle(phylo_structures_file)

        return phylo_structures_mat

    def _get_structure_file(self, homologous, version, mean=False):
        if mean:
            os.path.join(self._msa_data_path, f'{homologous}_mean_v{version}.pkl')
        return os.path.join(self._msa_data_path, f'{homologous}_structure_v{version}.pkl')

    def get_average_modeller_dm(self, n_structures=1):
        """Returns the average cm over few modeller predicted structures

        Args:
            n_structures (int): number of modeller predicted structures

        Returns:
            np.array: of shape (l, l)

        """

        if self._is_broken:
            return

        cms = []

        test_sets = set(DATASETS.pfam) | set(DATASETS.membrane) | set(DATASETS.cameo)

        if self.closest_reference is None or self.protein.target not in test_sets:
            return

        for n_struc in range(1, n_structures + 1):
            pdb_file_path = get_modeller_pdb_file(target=self.protein.target, n_struc=n_struc)
            has_modeller_file = os.path.isfile(pdb_file_path)
            if not has_modeller_file:
                self._run_modeller(n_structures=n_struc)
            modeller_prot = Protein(self.protein.protein,
                                    self.protein.chain,
                                    pdb_path=PATHS.modeller,
                                    modeller_n_struc=n_struc)
            modeller_dm = modeller_prot.dm
            assert len(modeller_prot.sequence) == len(self.protein.sequence)

            # thres = 8 if version == 4 else 8.5
            #
            cms.append(modeller_dm)

        return np.nanmean(np.stack(cms, axis=2), axis=2)

    def _get_cm(self, dm):
        if dm is None:
            return
        nan_inds = np.logical_or(np.isnan(dm), dm == self.NAN_VALUE)
        dm[nan_inds] = self.NAN_VALUE

        cm = (dm < self._THRESHOLD).astype(np.float32)

        cm[nan_inds] = self.NAN_VALUE

        return cm

    @property
    def reference_cm(self):
        return self._get_cm(self.reference_dm)

    @property
    def modeller_cm(self):
        return self._get_cm(self.modeller_dm)

    @property
    def target_pdb_dm(self):
        return self.protein.get_dist_mat(force=True)

    @property
    def target_pdb_cm(self):
        return self._get_cm(self.target_pdb_dm)

    @property
    def raptor_properties(self):
        msa = None if self._family is None else self.get_no_target_gaps_msa(sub=True)
        prop = get_properties(self.target, family=self._family, train=self._train, msa=msa)
        if prop is None:
            return
        if prop.shape[0] != self.seq_target.shape[0]:
            return
        return prop

    @property
    def ccmpred(self):

        ccmpred_mat_file = get_target_ccmpred_file(self.target, self._family)
        if not os.path.isfile(ccmpred_mat_file):
            self._run_ccmpred()

        ccmpred_mat = np.loadtxt(ccmpred_mat_file)
        if self._family is not None:
            target_seq_arr = self.target_seq_msa
            inds = np.where(target_seq_arr != '-')[0]
            row_idx = np.array(inds)
            col_idx = np.array(inds)
            ccmpred_mat_slice = ccmpred_mat[row_idx[:, None], col_idx]

            target_msa_seq_no_gaps = ''.join(target_seq_arr[inds])
            sub_ind = self.str_seq.find(target_msa_seq_no_gaps)
            rng = list(range(sub_ind, sub_ind + len(target_msa_seq_no_gaps)))
            l = self.seq_len
            ccmpred_mat = np.zeros(shape=(l, l))
            idx = np.array(rng)
            ccmpred_mat[idx[:, None], idx] = ccmpred_mat_slice

        return ccmpred_mat

    @property
    def evfold(self):
        # Gets the plmc evolutionary coupling array

        version = self._EVFOLD_VERSION

        evfold_file = '%s_v%s.txt' % (self.protein.target, version)
        evfold_mat_file = '%s_v%s.pkl' % (self.protein.target, version)
        evfold_path = os.path.join(get_target_path(self.target), 'evfold')
        check_path(evfold_path)

        evfold_mat_path = os.path.join(evfold_path, evfold_mat_file)

        if os.path.isfile(evfold_mat_path):
            evfold_mat = pkl_load(evfold_mat_path)
            if evfold_mat.shape[0] != len(self.protein.str_seq):
                self._run_evfold()
                return
            return evfold_mat

        pdb_seq = "".join(list(self.protein.sequence))

        def _get_sotred_ec(raw_ec):
            # gets the sorted indices for ec

            raw_ec_local = raw_ec.copy()

            ec_no_na = raw_ec_local.dropna()

            ec_no_dup = ec_no_na.drop_duplicates('i')
            ec_sorted_i = ec_no_dup.sort_values('i')
            ec_sorted_j = ec_no_dup.sort_values('j')

            return ec_sorted_i, ec_sorted_j, ec_no_na

        def _get_msa_seq_ec(raw_ec):
            # returns the msa sequence of the ec data

            raw_ec_local = raw_ec.copy()

            ec_sorted_i, ec_sorted_j, ec = _get_sotred_ec(raw_ec_local)

            # ec shows no duplicates hence i goes until on aa before the last
            last_msa_string = ec_sorted_j.A_j.iloc[-1]

            msa_seq = "".join(list(ec_sorted_i.A_i) + list(last_msa_string))
            return msa_seq

        ec_file = os.path.join(evfold_path, evfold_file)

        if not os.path.isfile(ec_file):
            self._run_evfold()

        raw_ec = read_raw_ec_file(ec_file)
        msa_seq = _get_msa_seq_ec(raw_ec)
        is_valid = msa_seq == pdb_seq

        if not is_valid:
            raise ValueError(
                'msa sequence must be a sub-string of pdb sequence'
                '\n\nMSA: %s\n\nPDB: %s\n\n' % (msa_seq, pdb_seq))

        ec_sorted_i, ec_sorted_j, ec = _get_sotred_ec(raw_ec)

        sorted_msa_indices = ec_sorted_i.i.to_list() + ec_sorted_j.j.to_list()
        sorted_msa_indices.sort()

        msa_ind_to_pdb_ind_map = {
            msa: pdb
            for pdb, msa in enumerate(set(sorted_msa_indices))
        }

        pdb_i = ec.loc[:, 'i'].map(msa_ind_to_pdb_ind_map).copy()
        pdb_j = ec.loc[:, 'j'].map(msa_ind_to_pdb_ind_map).copy()

        target_indices_pdb = (np.array(pdb_i).astype(np.int64),
                              np.array(pdb_j).astype(np.int64))

        l = len(self.protein.sequence)

        evfold = np.zeros(shape=(l, l))

        evfold[target_indices_pdb] = ec['cn'].values

        pkl_save(data=evfold, filename=evfold_mat_path)

        return evfold

    @property
    def reference_dm(self):
        reference = self.closest_reference
        dm_file = self._get_structure_file(reference, self._STRUCTURES_VERSION)
        if os.path.isfile(dm_file):
            return pkl_load(dm_file)
        return

    def _get_clustalo_msa(self):

        if hasattr(self, '_aln'):
            return

        clustalo_path = os.path.join(self._msa_data_path, 'clustalo')
        check_path(clustalo_path)
        fname = os.path.join(clustalo_path, f'aln_r_{self._n_refs}.fasta')
        for f in os.listdir(clustalo_path):
            full_p = os.path.join(clustalo_path, f)
            if full_p != fname:
                os.remove(full_p)
        if os.path.isfile(fname):
            # LOGGER.info(f"Reading {fname}")
            self._aln = read_fasta(fname, True)
            return

        msa_file = get_aln_fasta(self.target)
        if not os.path.isfile(msa_file):
            self._run_hhblits()

        structures = self._get_k_closest_references()
        if structures is None:
            return

        n_strucs = len(structures)

        structures = structures if self._n_refs >= n_strucs else structures[n_strucs - self._n_refs:n_strucs]
        ref_map = self.aligner.get_ref_map()
        templates_list = [ref_map[s] for s in structures]
        templates_list = [t for t in templates_list if len(Protein(t[0:4], t[4]).str_seq) > 0]

        def _get_record(t):
            prot = Protein(t[0:4], t[4])
            seq = prot.str_seq

            return SeqRecord(Seq(seq), id=t, name='', description='')

        aln_list = [self.target] + templates_list
        aln_list = [t for t in aln_list if Protein(t[0:4], t[4]).str_seq is not None]

        sequences = [_get_record(t) for t in aln_list]
        run_clustalo(sequences, fname, self.target, structures, self._family)

        aln = read_fasta(fname, True)

        self._aln = aln

    @property
    def templates_aln(self):
        return self.aligner.get_structures_msa()
        # self._get_clustalo_msa()
        # return self._aln

    def _replace_nas(self, array):
        return np.nan_to_num(array, nan=self.NAN_VALUE)

    @property
    def refs_contacts(self):
        dms = self.k_reference_dm_test
        if dms is None:
            return
        dms[np.logical_or(dms == -1, dms == 0)] = np.nan

        ref = np.array(np.nanmin(dms, 2) < self._THRESHOLD, dtype=np.int)
        return self._replace_nas(ref)

    @property
    def n_refs_test(self):
        self._get_clustalo_msa()
        n_strucs = len(self._aln) - 1
        return n_strucs

    @property
    def seq_len(self):
        return np.sum(self.target_seq_msa != '-') if self._family is not None else len(self.protein.str_seq)

    @property
    def k_reference_dm_test(self):

        out = self.trim_pad_arr(self.aligner.templates_distance_tensor)
        return_zeros = not self._require_template and out is None
        if not return_zeros:
            return out
        zeros = np.zeros((self.seq_len, self.seq_len, self._n_refs))
        return zeros

    def trim_pad_arr(self, arr):
        if arr is None:
            return
        n_strucs = int(arr.shape[-1])
        if n_strucs < self._n_refs:
            shape = list(arr.shape)
            shape[-1] = self._n_refs - n_strucs
            zero_array = np.zeros(shape)

            arr_out = np.concatenate([zero_array, arr], axis=2)
        if n_strucs >= self._n_refs:
            arr_out = arr[..., (n_strucs - self._n_refs): n_strucs]
        return self._replace_nas(arr_out)

    @property
    def seq_target(self):
        return self._get_seq_target()

    @property
    def seq_refs_test(self):
        # return self._get_seq_refs_test()
        out = self.trim_pad_arr(self.aligner.templates_sequence_tensor)
        return_zeros = not self._require_template and out is None
        if not return_zeros:
            return out
        l = self.seq_len
        zeros = np.zeros((l, 22, self._n_refs))
        return zeros

    @property
    def seq_refs_ss_acc(self):

        out = self.trim_pad_arr(self.aligner.templates_ss_acc_seq_tensor)
        return_zeros = not self._require_template and out is None
        if not return_zeros:
            return out
        l = self.seq_len
        zeros = np.zeros((l, 31, self._n_refs))
        return zeros

    @property
    def modeller_dm(self):

        if not self.has_refs:
            return

        pdb_file_path = get_modeller_pdb_file(target=self.protein.target, n_struc=1, templates=True)

        if not os.path.isfile(pdb_file_path):
            self.run_modeller_templates()
            if not os.path.isfile(pdb_file_path):
                return

        modeller_prot = Protein(self.protein.target[0:4],
                                self.protein.target[4],
                                pdb_path=PATHS.modeller,
                                version=self._STRUCTURES_VERSION)
        if modeller_prot.dm[0].shape == self.protein.dm.shape[0]:
            return modeller_prot.dm

        modeller_seq = self.protein.modeller_str_seq
        protein_seq = self.protein.str_seq
        aligned_seq_modeller = pairwise2.align.globalxx(modeller_seq, protein_seq, one_alignment_only=True)[0][0]
        dm = np.empty_like(self.protein.dm)
        dm[:] = np.nan
        inds_aln = np.where(np.array(list(aligned_seq_modeller)) != '-')[0]
        inds_mod = np.array(list(range(len(modeller_seq))))

        pairs_aln = list(itertools.combinations(inds_aln, 2))
        pairs_mod = list(itertools.combinations(inds_mod, 2))

        inds_aln = list(zip(*pairs_aln))
        inds_pdb = list(zip(*pairs_mod))

        mod_dm = modeller_prot.dm
        dm[inds_aln[0], inds_aln[1]] = mod_dm[inds_pdb[0], inds_pdb[1]]
        dm[inds_aln[1], inds_aln[0]] = mod_dm[inds_pdb[1], inds_pdb[0]]

        return dm

    @property
    def closest_pdb(self):
        return self.aligner.closest_template
        # ref_map = self.metadata['references_map'].get(self.closest_reference, None)
        # if ref_map is None:
        #     return
        # return ref_map[0][0]

    @property
    def n_homs(self):
        return self.aligner.n_homs
