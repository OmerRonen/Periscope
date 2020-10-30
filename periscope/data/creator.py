import collections
import itertools
import logging
import os
import subprocess
import tempfile
import urllib

import numpy as np
import pandas as pd

from scipy.special import softmax
from Bio import SeqIO, pairwise2
from Bio.Align import MultipleSeqAlignment, AlignInfo
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from ..utils.protein import Protein
from ..utils.constants import PATHS, DATASETS, AMINO_ACID_STATS, PROTEIN_BOW_DIM, SEQ_ID_THRES, N_REFS
from ..utils.utils import (convert_to_aln, write_fasta, MODELLER_VERSION, create_sifts_mapping, read_raw_ec_file,
                           pkl_save, pkl_load, compute_structures_identity_matrix, VERSION, get_modeller_pdb_file,
                           get_target_path, get_target_ccmpred_file, check_path, read_fasta, run_clustalo,
                           get_aln_fasta, get_predicted_pdb, save_chain_pdb)

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
    _PHYLO_VERSION = 5
    _THRESHOLD = 8

    def __init__(self, target, n_refs=N_REFS):
        self.protein = Protein(target[0:4], target[4])
        try:
            self._is_broken = not hasattr(self.protein, 'sequence')
        except Exception:
            self._is_broken = True
        self._msa_data_path = os.path.join(get_target_path(target), 'features')
        check_path(self._msa_data_path)
        self.target = target
        self._n_refs = n_refs
        self.metadata = self._get_metadata()
        self.has_refs = self.sorted_structures is not None
        self.has_refs_new = len(self._get_refs_aln()) > 1
        self.refactored = self.metadata['refactored']
        self.recreated = self.metadata.get('recreated', False)
        if not os.path.isfile(self.fasta_fname):
            self._write_fasta()

    @property
    def msa_length(self):
        return len(self._parse_msa())

    def generate_data(self):
        if self._is_broken or self.recreated:
            return

        self._run_hhblits()
        msa = self._parse_msa()
        self.evfold
        self.ccmpred
        self._find_all_references(msa)
        self.k_reference_dm_new
        LOGGER.info('Generating phylo dm')
        self._get_phylo_structures_mat()
        LOGGER.info('Sorting structures')
        self._get_sorted_structures()
        LOGGER.info('Saving metadata')
        self.metadata['recreated'] = True
        self._save_metadata()

    def recreate_data(self):

        if not self.metadata['refactored']:
            LOGGER.info('Cleaning folder')
            self._clean_target_folder()
            LOGGER.info(f'Finding references for {self.target}')
            self._find_all_references(self._parse_msa())
            LOGGER.info('Generating phylo dm')
            self._get_phylo_structures_mat()
            LOGGER.info('Sorting structures')
            self._get_sorted_structures()
            LOGGER.info('Saving metadata')
            self.metadata['refactored'] = True
            self._save_metadata()

    def recreate_data_after_find(self):
        LOGGER.info('Generating phylo dm')
        self._get_phylo_structures_mat()
        LOGGER.info('Sorting structures')
        self._get_sorted_structures()
        LOGGER.info('Saving metadata')
        self.metadata['refactored'] = True
        self._save_metadata()

    @property
    def sorted_structures(self):
        structures_sorted_file = os.path.join(
            self._msa_data_path, 'structures_sorted_2.pkl')

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

    def _get_no_target_gaps_msa(self):

        msa_file = get_aln_fasta(self.target)

        # msa_file = os.path.join(get_target_path(self.target), 'hhblits', self.target + '_v%s.fasta' % version)

        if not os.path.isfile(msa_file):
            self._run_hhblits()

        # full_alphabet = Gapped(ExtendedIUPACProtein())
        # fasta_seqs = [f.upper() for f in list(SeqIO.parse(msa_file, "fasta", alphabet=full_alphabet))]
        fasta_seqs = [f.upper() for f in list(SeqIO.parse(msa_file, "fasta"))]
        target_seq_full = fasta_seqs[0].seq
        target_seq_no_gap_inds = [i for i in range(len(target_seq_full)) if target_seq_full[i] != '-']
        target_seq = ''.join(target_seq_full[i] for i in target_seq_no_gap_inds)

        def _slice_seq(seq, inds):
            seq.seq = Seq(''.join(seq.seq[i] for i in inds))
            return seq

        assert target_seq == "".join(self.protein.sequence)
        fasta_seqs_short = [_slice_seq(s, target_seq_no_gap_inds) for s in fasta_seqs]
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
        templates_list = [self.metadata['references_map'][s][0][0] for s in structures]
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
        templates_list = [self.metadata['references_map'][s][0][0] for s in structures]
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

        reference = self.metadata['references_map'][self.closest_reference][0][0]

        if reference is None:
            LOGGER.info('Reference not found for %s' % self.target)
            return

        template_protein, template_chain = reference[0:4], reference[4]

        mod_args = f'{self.protein.protein}  {self.protein.chain} {template_protein} {template_chain} {n_structures}'

        # saving target in pir format
        target_seq = SeqIO.SeqRecord(Seq(''.join(self.protein.sequence)),
                                     name='sequence:%s:::::::0.00: 0.00' % self.target, id=self.target, description='')
        # SeqIO.PirIO.PirWriter(open(os.path.join(periscope_path, '%s.ali'%self.target), 'w'))
        SeqIO.write(target_seq, os.path.join(os.getcwd(), '%s.ali' % self.target), 'pir')
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

        msa = self._get_no_target_gaps_msa()

        with tempfile.NamedTemporaryFile() as tmp:
            write_fasta(msa, tmp.name)

            with tempfile.NamedTemporaryFile(suffix='.aln') as tmp2:
                convert_to_aln(tmp.name, tmp2.name)

                ccmpred_mat_file = get_target_ccmpred_file(self.target)

                ccmpred_cmd = ['ccmpred', tmp2.name, ccmpred_mat_file]

                p = subprocess.run(ccmpred_cmd)
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
        hhblits_path = os.path.join(get_target_path(self.target), 'hhblits')
        file_msa = os.path.join(hhblits_path, '%s_v%s.fasta' % target_version)

        if not os.path.isfile(file_msa):
            self._run_hhblits()

        cmd = f'plmc -o {file_params} -c {file_ec} -le 16.0 -m {500} -lh 0.01  -g -f {self.target} {file_msa}'

        subprocess.call(cmd, shell=True)

    def _reformat_old_file(self):
        version = self._MSA_VERSION

        old_a2m_file = os.path.join(PATHS.msa, 'hhblits', 'a2m', f'{self.target}.a2m')
        if not os.path.isfile(old_a2m_file):
            return False
        target_hhblits_path = os.path.join(get_target_path(self.target), 'hhblits')
        check_path(target_hhblits_path)
        fasta_file = os.path.join(target_hhblits_path, self.target + '_v%s.fasta' % version)

        a3m_file = os.path.join(target_hhblits_path, self.target + '.a3m')

        reformat = ['reformat.pl', old_a2m_file, a3m_file]
        subprocess.run(reformat)

        reformat = ['reformat.pl', old_a2m_file, fasta_file]
        subprocess.run(reformat)
        return True

    @property
    def fasta_fname(self):
        return os.path.join(get_target_path(self.target), self.target + '.fasta')

    def _write_fasta(self):
        if self.protein.str_seq is None:
            return
        seq = Seq(self.protein.str_seq.upper())
        sequence = SeqIO.SeqRecord(seq, name=self.target, id=self.target)
        query = os.path.join(get_target_path(self.target), self.target + '.fasta')
        SeqIO.write(sequence, query, "fasta")

    def _run_hhblits(self):
        # Generates multiple sequence alignment using hhblits
        if self.protein.str_seq is None:
            return
        seq = Seq(self.protein.str_seq.upper())

        target = self.target
        version = self._MSA_VERSION

        sequence = SeqIO.SeqRecord(seq, name=target, id=target)
        target_hhblits_path = os.path.join(get_target_path(target), 'hhblits')
        check_path(target_hhblits_path)

        query = os.path.join(target_hhblits_path, target + '.fasta')
        SeqIO.write(sequence, query, "fasta")
        output_hhblits = os.path.join(target_hhblits_path, target + '.a3m')
        output_reformat1 = os.path.join(target_hhblits_path, target + '.a2m')
        output_reformat2 = os.path.join(target_hhblits_path, target + '_v%s.fasta' % version)

        db_hh = '/vol/sci/bio/data/or.zuk/projects/ContactMaps/data/uniprot20_2016_02/uniprot20_2016_02'

        hhblits_params = '-n 3 -e 1e-3 -maxfilt 10000000000 -neffmax 20 -nodiff -realign_max 10000000000'

        hhblits_cmd = f'hhblits -i {query} -d {db_hh} {hhblits_params} -oa3m {output_hhblits}'

        subprocess.run(hhblits_cmd, shell=True, stdout=open(os.devnull, 'wb'))
        reformat = ['reformat.pl', output_hhblits, output_reformat1]
        subprocess.run(reformat)

        reformat = ['reformat.pl', output_reformat1, output_reformat2]
        subprocess.run(reformat)

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

    def _parse_msa(self):
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
            return seq.id.split('|')[1]

        # fasta_seqs = list(SeqIO.parse(msa_file, "fasta", alphabet=Gapped(ExtendedIUPACProtein())))
        fasta_seqs = list(SeqIO.parse(msa_file, "fasta"))
        sequences = {
            _get_id(seq): SeqRecord(seq.seq.upper(),
                                    id=_get_id(seq).split("_")[0])
            for seq in fasta_seqs
        }

        return sequences

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
            return np.squeeze(pkl_load(seq_target_file))

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

    def _get_seq_refs_full(self):
        """Numeric sequence representation of references

        Returns:
            np.array: of shape (l, PROTEIN_BOW_DIM, n_structures)

        """

        seq_refs_file = os.path.join(self._msa_data_path, 'refs_seqs.pkl')

        if os.path.isfile(seq_refs_file):
            return pkl_load(seq_refs_file)

        return self._generate_seq_refs_full()

    def _generate_seq_refs_full_test(self):

        refs = self._get_k_closest_references()

        if refs is None:
            return

        self._get_clustalo_msa()

        msa = {s.id: s for s in self._aln}

        bow_msa_refs = self._bow_msa(msa=msa).transpose(1, 2, 0)
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

        file = os.path.join(self._msa_data_path, 'seq_refs_test.pkl')
        if os.path.isfile(file):
            output = pkl_load(file)
            if output.shape[2] == self._n_refs:
                return output

        bow_msa_full = self._generate_seq_refs_full_test()

        if bow_msa_full is None:
            return

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

    def _get_seq_refs(self):
        """Numeric sequence representation of references

        Returns:
            np.array: of shape (l, PROTEIN_BOW_DIM, k)

        """

        bow_msa_full = self._get_seq_refs_full()

        if bow_msa_full is None:
            return

        shape = bow_msa_full.shape
        total_refs = shape[-1]

        if total_refs >= self._n_refs:
            return bow_msa_full[..., 0:self._n_refs]
        else:
            return np.concatenate(
                [np.zeros((shape[0], shape[1], self._n_refs - total_refs)), bow_msa_full],
                axis=2)

    @property
    def closest_reference(self):

        sorted_structures = self.sorted_structures
        if sorted_structures is None:
            return
        structure = sorted_structures.index[0]
        assert structure != self.protein.target
        return structure

    def _get_pssm(self):
        """Pssm array

        Returns:
            np.array: the msa pssm of shape (l, 26)

        """

        version = self._MSA_VERSION

        pssm_file = os.path.join(self._msa_data_path, 'pssm_bio_v%s.pkl' % version)

        if os.path.isfile(pssm_file):
            return pkl_load(pssm_file)

        msa_file = os.path.join(PATHS.hhblits, 'fasta',
                                self.target + '_v%s.fasta' % version)

        # full_alphabet = Gapped(ExtendedIUPACProtein())

        fasta_seqs = [f.upper() for f in list(SeqIO.parse(msa_file, "fasta"))]
        target_seq_full = fasta_seqs[0].seq
        target_seq_no_gap_inds = [i for i in range(len(target_seq_full)) if target_seq_full[i] != '-']
        target_seq = ''.join(target_seq_full[i] for i in target_seq_no_gap_inds)

        def _slice_seq(seq, inds):
            seq.seq = Seq(''.join(seq.seq[i] for i in inds))
            return seq

        assert target_seq == "".join(self.protein.sequence)
        fasta_seqs_short = [_slice_seq(s, target_seq_no_gap_inds) for s in fasta_seqs]
        msa = MultipleSeqAlignment(fasta_seqs_short)
        summary = AlignInfo.SummaryInfo(msa)

        # short_alphabet = IUPACProtein()
        # chars_to_ignore = list(set(full_alphabet.letters).difference(set(short_alphabet.letters)))

        n_homologous = len(msa)

        expected = collections.OrderedDict(sorted(AMINO_ACID_STATS.items()))
        expected = {k: n_homologous * v / 100 for k, v in expected.items()}
        pssm = summary.pos_specific_score_matrix()
        pssm = np.array([list(p.values()) for p in pssm]) / np.array(list(expected.values()))[:, None].T

        epsilon = 1e-06

        pssm_log = -1 * np.log(np.clip(pssm, a_min=epsilon, a_max=None))

        pkl_save(pssm_file, pssm_log)

        return pssm

    def _get_aligned_ss(self):

        ss_refs_file = os.path.join(self._msa_data_path, 'ss_refs.pkl')
        if os.path.isfile(ss_refs_file):
            return pkl_load(ss_refs_file)

        return self._generate_aligned_ss()

    def _generate_aligned_ss(self):

        ss_refs_file = os.path.join(self._msa_data_path, 'ss_refs.pkl')

        refs = self._get_k_closest_references()
        if refs is None:
            return

        parsed_msa = self._parse_msa()
        target_seq_msa = parsed_msa[self.target].seq
        valid_inds = [i for i in range(len(target_seq_msa)) if target_seq_msa[i] != '-']
        parsed_msa = {r: parsed_msa[r].seq for r in refs}
        na_arr = np.array([0, 0, 0, 0, 0, 0, 0, 1])
        secondary_structures = {r: np.stack([na_arr] * len(target_seq_msa), axis=0) for r in
                                refs}
        for ref, seq in parsed_msa.items():
            reference = self.metadata['references_map'][ref][0][0]
            start_ind_pdb, end_ind_pdb = self.metadata['references_map'][ref][0][1]
            ref_prot = Protein(reference[0:4], reference[4])
            reference_sequence_full = "".join(ref_prot.sequence)
            reference_sequence = reference_sequence_full[start_ind_pdb:end_ind_pdb]

            pdb_inds, msa_inds = self._align_pdb_msa(reference_sequence,
                                                     "".join(seq).upper(),
                                                     list(range(start_ind_pdb, end_ind_pdb)),
                                                     one_d=True)
            ss = ref_prot.secondary_structure
            secondary_structures[ref][msa_inds, :] = ss[pdb_inds, :]
            secondary_structures[ref] = secondary_structures[ref][valid_inds, :]

        aligned_ss = np.stack([secondary_structures[s] for s in refs], axis=2)
        pkl_save(ss_refs_file, aligned_ss)
        return aligned_ss

    @staticmethod
    def _align_pdb_msa(pdb_sequence, msa_sequence, pdb_indices, one_d=False):
        """Get alined indices if exists

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
        return list(self.metadata['references_map'].keys())

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

    def _find_all_references(self, msa):
        # finds and saves all known strctures in an alignment

        target_msa_seq = "".join(msa[self.target].seq)

        target_sequence = ''.join(self.protein.sequence)
        target_pdb_indices = list(range(0, len(target_sequence)))

        pdb_inds_target, msa_inds_target = self._align_pdb_msa(target_sequence,
                                                               target_msa_seq,
                                                               target_pdb_indices)
        for homologous in msa:
            aligned_structure = self._find_reference(uniprot_name=homologous,
                                                     msa_sequence=msa[homologous],
                                                     pdb_inds_target=pdb_inds_target,
                                                     msa_inds_target=msa_inds_target)

            if aligned_structure is None:
                continue

            pkl_save(self._get_structure_file(homologous, self._STRUCTURES_VERSION), aligned_structure['dm'])
            pkl_save(self._get_structure_file(homologous, self._STRUCTURES_VERSION, True), aligned_structure['dm_mean'])
            self.metadata['references_map'][homologous] = aligned_structure['map']

    def fix_capital_chain(self):

        def _is_fix_required(hom):
            if hom == self.target:
                return False
            required = False
            for pdb in self.metadata['references_map'][hom]:
                required |= pdb[0][4].upper() != pdb[0][4]

            return required

        requires_fix = [hom for hom in self.metadata['references_map'] if _is_fix_required(hom)]

        if len(requires_fix) == 0:
            return

        msa = self._parse_msa()
        target_msa_seq = "".join(msa[self.target].seq)

        target_sequence = ''.join(self.protein.sequence)
        target_pdb_indices = list(range(0, len(target_sequence)))

        pdb_inds_target, msa_inds_target = self._align_pdb_msa(target_sequence,
                                                               target_msa_seq,
                                                               target_pdb_indices)

        for homologous in requires_fix:
            aligned_structure = self._find_reference(uniprot_name=homologous,
                                                     msa_sequence=msa[homologous],
                                                     pdb_inds_target=pdb_inds_target,
                                                     msa_inds_target=msa_inds_target)

            if aligned_structure is None:
                del self.metadata['references_map'][homologous]
                self._save_metadata()
                self._generate_phylo_structures_mat()
                self._get_sorted_structures()
                self._generate_seq_refs_full()
                self._generate_aligned_ss()
                return

            pkl_save(self._get_structure_file(homologous, self._STRUCTURES_VERSION), aligned_structure['dm'])
            pkl_save(self._get_structure_file(homologous, self._STRUCTURES_VERSION, True), aligned_structure['dm_mean'])
            self.metadata['references_map'][homologous] = aligned_structure['map']

    def _find_reference(self, uniprot_name, msa_sequence, pdb_inds_target, msa_inds_target):

        has_mapping = uniprot_name in SIFTS_MAPPING
        if has_mapping:
            LOGGER.info(SIFTS_MAPPING[uniprot_name])
        is_target = uniprot_name == self.target

        if is_target:
            return {'dm': self.protein.dm, 'dm_mean': self.protein.dm, 'map': None}

        if not has_mapping and not is_target:
            return None

        msa_sequence = str(msa_sequence.seq)

        potential_structures = SIFTS_MAPPING[uniprot_name]

        aligned_potential_structures = self._align_potential_structures(potential_structures, msa_sequence,
                                                                        pdb_inds_target, msa_inds_target)

        if aligned_potential_structures is None:
            return

        dms = aligned_potential_structures['dms']

        dm = dms[0]
        dm_mean = np.nanmean(np.stack(dms, axis=2), axis=2)

        return {'dm': dm, 'dm_mean': dm_mean, "map": aligned_potential_structures['ref_map']}

    def _find_aligned_dm(self, protein_map, msa_sequence, pdb_inds_target, msa_inds_target):
        # this function inspects a single protein map and returns an aligned distance matrix if possible

        protein = protein_map[0][0:4]
        chain = protein_map[0][4]

        if protein == self.target[0:4]:
            return

        msa_distance_matrix = np.empty((len(msa_sequence), len(msa_sequence)))
        msa_distance_matrix[:] = np.nan

        reference_aligned_dm = np.empty((len(self.protein.sequence), len(self.protein.sequence)))
        reference_aligned_dm[:] = np.nan
        try:
            reference_protein = Protein(protein, chain)
            reference_sequence_full = reference_protein.modeller_str_seq
        except Exception:
            return

        start_ind_pdb = min(protein_map[1]['pdb'][0] - 1,
                            len(reference_sequence_full))
        end_ind_pdb = min(protein_map[1]['pdb'][1] - 1,
                          len(reference_sequence_full))

        reference_sequence = reference_sequence_full[start_ind_pdb:end_ind_pdb]

        pdb_inds, msa_inds = self._align_pdb_msa(pdb_sequence=reference_sequence,
                                                 msa_sequence=msa_sequence.upper(),
                                                 pdb_indices=list(range(start_ind_pdb, end_ind_pdb)))

        if pdb_inds is None:
            return

        msa_distance_matrix[msa_inds[0], msa_inds[1]] = reference_protein.dm[pdb_inds[0], pdb_inds[1]]

        reference_aligned_dm[pdb_inds_target[0],
                             pdb_inds_target[1]] = msa_distance_matrix[msa_inds_target[0],
                                                                       msa_inds_target[1]]
        reference_aligned_dm[pdb_inds_target[1],
                             pdb_inds_target[0]] = msa_distance_matrix[msa_inds_target[0],
                                                                       msa_inds_target[1]]

        return {'dm': reference_aligned_dm, 'reference': (reference_protein.target, (start_ind_pdb, end_ind_pdb))}

    def _align_potential_structures(self, potential_structures, msa_sequence, pdb_inds_target,
                                    msa_inds_target):

        aligned_dms = []
        references_map = []

        for protein_map in potential_structures:
            map = self._find_aligned_dm(protein_map, msa_sequence, pdb_inds_target, msa_inds_target)

            if map is None:
                continue

            aligned_dms.append(map['dm'])
            references_map.append(map['reference'])

        if len(aligned_dms) == 0:
            return

        return {'dms': aligned_dms, "ref_map": references_map}

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

    def _get_refs_ss(self):
        """Numeric sequence representation of references

        Returns:
            np.array: of shape (l, PROTEIN_BOW_DIM, k)

        """

        k = self._n_refs

        ss_refs_full = self._get_aligned_ss()

        if ss_refs_full is None:
            return

        shape = ss_refs_full.shape
        total_refs = shape[-1]

        if total_refs >= k:
            return ss_refs_full[..., 0:k]
        else:
            return np.concatenate(
                [np.zeros((shape[0], shape[1], k - total_refs)), ss_refs_full],
                axis=2)

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
    def ccmpred(self):

        ccmpred_mat_file = get_target_ccmpred_file(self.target)
        if os.path.isfile(ccmpred_mat_file):
            ccmpred_mat = np.loadtxt(ccmpred_mat_file)
            if ccmpred_mat.shape[0] != len(self.protein.str_seq):
                self._run_ccmpred()
                return
            return ccmpred_mat

        success = self._run_ccmpred()
        if success:
            return np.loadtxt(ccmpred_mat_file)

        return

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

    def _get_refs_aln(self):

        clustalo_path = os.path.join(get_target_path(self.target), 'clustalo')
        check_path(clustalo_path)
        fname = os.path.join(clustalo_path, f'aln_refs.fasta')

        if os.path.isfile(fname):
            return read_fasta(fname, True)

        msa = self._parse_msa()
        if msa is None:
            return []
        refs = []
        profile = []

        def get_pdb_ref(h):
            if h not in SIFTS_MAPPING or h == self.target:
                return

            pdbs = [map[0] for map in SIFTS_MAPPING[h]]
            if self.target in pdbs:
                return
            return pdbs[0]

        for h in msa:
            pdb_ref = get_pdb_ref(h)
            if pdb_ref is not None:
                refs.append(pdb_ref)
                profile.append(h)

        def _get_record(t):
            prot = Protein(t[0:4], t[4])
            seq = prot.str_seq

            return SeqRecord(Seq(seq), id=t, name='', description='')

        aln_list = [self.target] + refs
        aln_list = [t for t in aln_list if Protein(t[0:4], t[4]).str_seq is not None]

        sequences = [_get_record(t) for t in aln_list]

        run_clustalo(sequences, fname, self.target, profile)
        self._filter_aln(fname)
        aln = read_fasta(fname, True)
        return aln

    def _filter_aln(self, aln_fname):

        aln = read_fasta(aln_fname, True)

        seqs_to_filter = []
        s_t = [s for s in aln if s.id == self.target][0]
        target_s_arr = np.array(list(s_t.seq))
        valid_inds = target_s_arr != '-'
        target_s_arr_l = np.sum(valid_inds)

        def _seq_id(s):
            s_arr = np.array(list(s.seq))
            s_arr_l = len(s_arr[np.logical_and(s_arr != '-', valid_inds)])

            seq_id = np.sum(target_s_arr[valid_inds] == s_arr[valid_inds]) / min(s_arr_l, target_s_arr_l)
            return seq_id

        for s in aln:
            if s.id == self.target:
                continue

            seq_id = _seq_id(s)
            if seq_id > SEQ_ID_THRES:
                seqs_to_filter.append(s.id)
        aln = sorted([s for s in aln if s.id not in seqs_to_filter], key=_seq_id, reverse=True)
        SeqIO.write(aln, aln_fname, 'fasta')

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

        templates_list = [self.metadata['references_map'][s][0][0] for s in structures]
        templates_list = [t for t in templates_list if len(Protein(t[0:4], t[4]).str_seq) > 0]

        def _get_record(t):
            prot = Protein(t[0:4], t[4])
            seq = prot.str_seq

            return SeqRecord(Seq(seq), id=t, name='', description='')

        aln_list = [self.target] + templates_list
        aln_list = [t for t in aln_list if Protein(t[0:4], t[4]).str_seq is not None]

        sequences = [_get_record(t) for t in aln_list]
        run_clustalo(sequences, fname, self.target, structures)

        aln = read_fasta(fname, True)

        self._aln = aln

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

    @staticmethod
    def _get_aln_dm(seq, target):
        dm = np.empty((len(seq), len(seq)))
        dm[:] = np.nan
        inds_aln = np.where(np.array(list(seq)) != '-')[0]
        inds_pdb = np.array(list(range(len(inds_aln))))

        pairs_aln = list(itertools.combinations(inds_aln, 2))
        pairs_pdb = list(itertools.combinations(inds_pdb, 2))

        inds_aln = list(zip(*pairs_aln))
        inds_pdb = list(zip(*pairs_pdb))

        prot_dm = Protein(target[0:4], target[4]).dm
        dm[inds_aln[0], inds_aln[1]] = prot_dm[inds_pdb[0], inds_pdb[1]]
        dm[inds_aln[1], inds_aln[0]] = prot_dm[inds_pdb[1], inds_pdb[0]]

        return dm

    @property
    def n_refs_test(self):
        self._get_clustalo_msa()
        n_strucs = len(self._aln) - 1
        return n_strucs

    @property
    def n_refs_new(self):
        aln = self._get_refs_aln()
        n_strucs = len(aln) - 1
        return n_strucs

    @property
    def n_homs(self):

        return len(self._parse_msa())

    @property
    def k_reference_dm_new(self):

        f_name = os.path.join(self._msa_data_path, 'aligned_refs_new.pkl')
        aln = self._get_refs_aln()
        n_strucs = len(aln) - 1
        if n_strucs == 0:
            return
        if os.path.isfile(f_name):
            aligned_dms = pkl_load(f_name)

        else:

            s_target = aln[0]

            target_inds = np.where(np.array(list(s_target.seq)) != '-')[0]

            dms = np.stack([self._get_aln_dm(s.seq, s.id) for s in aln[1:]], axis=2)
            aligned_dms = dms[target_inds, :][:, target_inds]

            pkl_save(filename=f_name, data=aligned_dms)

        aligned_dms = aligned_dms[..., 0:self._n_refs]

        if n_strucs < self._n_refs:
            shape = (aligned_dms.shape[0], aligned_dms.shape[1], self._n_refs - n_strucs)
            zero_array = np.zeros(shape)

            aligned_dms = np.concatenate([zero_array, aligned_dms], axis=2)

        return self._replace_nas(aligned_dms)

    @property
    def k_reference_dm_test(self):

        file = os.path.join(self._msa_data_path, 'k_dm_tst.pkl')
        if os.path.isfile(file):
            output = pkl_load(file)
            if output.shape[2] == self._n_refs:
                return output

        structures = self._get_k_closest_references()
        if structures is None:
            return

        self._get_clustalo_msa()
        aln = self._aln
        dms_fname = os.path.join(self._msa_data_path, 'dms_test.pkl')
        if not os.path.isfile(dms_fname):
            try:
                dms = np.stack([self._get_aln_dm(s.seq, s.id) for s in aln[1:]], axis=2)
                pkl_save(data=dms, filename=dms_fname)
            except ValueError:
                return
        else:
            dms = pkl_load(dms_fname)
        target_inds = np.where(np.array(list(aln[0].seq)) != '-')[0]

        aligned_dms = dms[target_inds, :][:, target_inds]
        n_strucs = aligned_dms.shape[-1]

        if n_strucs < self._n_refs:
            shape = (aligned_dms.shape[0], aligned_dms.shape[1], self._n_refs - n_strucs)
            zero_array = np.zeros(shape)

            aligned_dms = np.concatenate([zero_array, aligned_dms], axis=2)
        if n_strucs > self._n_refs:
            aligned_dms = aligned_dms[..., n_strucs - self._n_refs: n_strucs]

        output = self._replace_nas(aligned_dms)
        pkl_save(file, output)
        return output

    @property
    def k_reference_dm(self):
        dms = []
        structures = self._get_k_closest_references()

        if structures is None:
            return

        n_strucs = len(structures)

        structures = structures if self._n_refs >= n_strucs else structures[n_strucs - self._n_refs:n_strucs]

        for i in range(min(self._n_refs, n_strucs)):
            s = structures[i]
            dm = pkl_load(os.path.join(self._get_structure_file(s, self._STRUCTURES_VERSION)))
            dms.append(dm)

        if n_strucs < self._n_refs:
            zero_array = np.zeros_like(dms[0])
            masking = [zero_array] * (self._n_refs - n_strucs)
            dms = masking + dms

        return self._replace_nas(np.stack(dms, axis=2))

    @property
    def seq_target(self):
        return self._get_seq_target()

    @property
    def seq_refs_test(self):
        return self._get_seq_refs_test()

    @property
    def seq_refs(self):
        return self._get_seq_refs()

    @property
    def seq_refs_pssm(self):
        refs_pssm = [self.seq_refs, np.repeat(np.expand_dims(self._get_pssm(), axis=2), self._n_refs, 2)]
        return np.array(np.concatenate(refs_pssm, axis=1), dtype=np.float32)

    @property
    def seq_target_pssm(self):
        return np.array(np.concatenate([self.seq_target, self._get_pssm()], axis=1), dtype=np.float32)

    @property
    def seq_refs_ss(self):
        ss_refs = self._get_refs_ss()

        seq_refs_ss = np.concatenate([ss_refs, self.seq_refs], axis=1) if ss_refs is not None else None

        return seq_refs_ss

    @property
    def seq_target_ss(self):
        has_ss = self.protein.secondary_structure is not None

        seq_target_ss = np.concatenate([self.protein.secondary_structure, self.seq_target], axis=1) if has_ss else None

        return seq_target_ss

    @property
    def k_reference_dm_conv(self):
        if self.k_reference_dm is None:
            return
        return np.squeeze(self.k_reference_dm)

    @property
    def seq_target_pssm_ss(self):
        has_ss = self.protein.secondary_structure is not None

        ss_pssm = [self.protein.secondary_structure, self.seq_target_pssm]
        seq_target_pssm_ss = np.concatenate(ss_pssm, axis=1) if has_ss else None

        return seq_target_pssm_ss

    @property
    def seq_refs_pssm_ss(self):
        ss_refs = self._get_refs_ss()

        seq_refs_pssm_ss = np.concatenate([ss_refs, self.seq_refs_pssm], axis=1) if ss_refs is not None else None

        return seq_refs_pssm_ss

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
        ref_map = self.metadata['references_map'].get(self.closest_reference, None)
        if ref_map is None:
            return
        return ref_map[0][0]
