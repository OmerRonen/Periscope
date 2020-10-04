class MSA():
    def __init__(self,
                 target,
                 generate_msa=False,
                 calculate_plmc=False,
                 version=_VERSION):
        LOGGER.info("MSA of %s" % target)
        self.target = target
        self.protein_name = target[0:4]
        self.chain_name = target[4]
        self.protein = Protein(self.protein_name, self.chain_name)
        self.angels_true = self.protein.angels

        self.msa_data_path = path.join(MSA_STRUCTURES_DATA_PATH, self.target)
        self.metadata_fname = path.join(self.msa_data_path, 'metadata.pkl')

        _metadata = {'version': version, 'structures': {}}
        has_metadata = os.path.isfile(self.metadata_fname)
        data_version = 0
        if has_metadata:
            m_data = pkl_load(self.metadata_fname)
            data_version = m_data.get('version', 0)

        is_new = data_version > 1 if has_metadata else False
        self.metadata = pkl_load(self.metadata_fname) if is_new else _metadata
        self.version = self.metadata['version']

        if not path.exists(self.msa_data_path):
            mkdir(self.msa_data_path)

        self.msa_filename = os.path.join(
            HHBLITS_PATH, 'fasta',
            '%s_v%s.fasta' % (self.target, self.version))

        if not path.exists(self.msa_filename) or generate_msa:
            self.hhblits_msa(version=self.version)

        self.data = read_fasta(self.msa_filename, True)
        self.target_pdb_dm = self.protein.dm
        self.target_pdb_cm = (self.target_pdb_dm < THRESHOLD).astype(
            np.float32)
        self.target_angels = np.nan_to_num(self.protein.angels, 0)

        self.target_seq = self.protein.sequence
        self.target_seq_length_pdb = len(self.target_seq)
        if self.target not in self.data:
            print('Target %s not in msa data' % self.target)
        else:
            self.target_seq_length_msa = len(self.data[self.target]['seq'])
            self.run_plmc(force=calculate_plmc, version=self.version)
            self.read_plmc_data(version=self.version)
            self.known_structures = set(self.metadata['structures'].keys())
            refind = len([
                s for s in list(self.metadata['structures'].values())
                if s == self.target
            ]) > 1
            if len(self.metadata['structures']) == 0 or refind:
                self.find_all_structures()
                self.load_known_structures()
            else:
                self.load_known_structures()
            self.structures_dist_mat()
        struc = self.closest_known_strucutre
        dm = self.data[struc]['dist mat pdb']
        reference_nan = np.isnan(dm)
        reference_dm = dm
        reference_cm = (reference_dm < THRESHOLD).astype(np.float32)

        reference_dm[reference_nan] = -1
        reference_cm[reference_nan] = -1

        self.reference_cm = np.expand_dims(reference_cm, axis=2)

        self.reference_dm = np.expand_dims(reference_dm, axis=2)
        # self.loaded = False
        # if path.exists(self.msa_data_path):
        #     self.load_known_structures()
        # else:
        #     mkdir(self.msa_data_path)
        #     sifts_mapping = create_sifts_mapping()
        #     self.find_all_structures(sifts_mapping)
        #
        # self.structures_dist_mat()
        #
        # self.aa_freq_df = pd.DataFrame({'A': 9.23, 'Q': 3.77, 'L': 9.90, 'S': 6.63,
        #                            'R': 5.77, 'E': 6.16, 'K': 4.90, 'T': 5.55,
        #                            'N': 3.81, 'G': 7.35, 'M': 2.37, 'W': 1.30,
        #                            'D': 5.48, 'H': 2.19, 'F': 3.91, 'Y': 2.90,
        #                            'C': 1.19, 'I': 5.65, 'P': 4.87, 'V': 6.92, 'X': 0.04}, index=['frequency'])/100
        # self._features = {}

    @timefunc
    def hhblits_msa(
        self,
        msa_path='/cs/zbio/orzuk/projects/ContactMaps/data/MSA_Completion/msa_data',
        version=2):
        target_protein = self.protein_name
        chain = self.chain_name
        protein_name = target_protein + chain
        sequence = Seq(''.join(self.protein.sequence))
        sequence = SeqIO.SeqRecord(sequence,
                                   name=protein_name,
                                   id=protein_name)
        query = path.join(msa_path, 'query', protein_name + '.fasta')
        SeqIO.write(sequence, query, "fasta")
        output_hhblits = path.join(msa_path, 'hhblits', 'a3m',
                                   protein_name + '.a3m')
        output_reformat1 = path.join(msa_path, 'hhblits', 'a2m',
                                     protein_name + '.a2m')
        output_reformat2 = path.join(msa_path, 'hhblits', 'fasta',
                                     protein_name + '_v%s.fasta' % version)

        db_hh = '/cs/zbio/orzuk/projects/ContactMaps/data/MSA_Completion/hh/uniprot20_2016_02/uniprot20_2016_02'

        hhblits = [
            'hhblits', '-i', query, '-d', db_hh, '-n', '3', '-e', '1e-3',
            '-maxfilt', '10000000000', '-neffmax', '20', '-nodiff', '-realign',
            '-realign_max', '10000000000', '-oa3m', output_hhblits
        ]
        subprocess.call(hhblits)
        reformat = ['reformat.pl', output_hhblits, output_reformat1]
        subprocess.call(reformat)

        # output_hhfilter = path.join(msa_path, 'hhblits', 'filltered', protein_name + '_fill.a3m')
        #
        # hhfilter = ['hhfilter', '-i', output_reformat1, '-M', '25', '-cov', '75', '-o', output_hhfilter]
        # call(hhfilter)

        reformat = ['reformat.pl', output_reformat1, output_reformat2]
        subprocess.call(reformat)

    def load_known_structures(self):
        structures_files = [
            '%s_pdb.pkl' % s for s in self.metadata['structures']
        ]
        angels_files = [
            '%s_angels.pkl' % s for s in self.metadata['structures']
        ]

        for struc_file in structures_files:
            s = struc_file.split('_')[0]
            if self.metadata['structures'][s][
                    0:4] == self.protein_name and s != self.target:
                continue
            pdb_dist_mat_file = path.join(self.msa_data_path, struc_file)
            self.data[s]['dist mat pdb'] = pkl_load(pdb_dist_mat_file)
            self.data[s]['dist mat pdb'][self.data[s]['dist mat pdb'] ==
                                         -1] = np.nan
            self.known_structures.add(s)

        for angels_file in angels_files:
            angels_full_path = path.join(self.msa_data_path, angels_file)
            s = angels_file.split('_')[0]
            if self.metadata['structures'][s][
                    0:4] == self.protein_name and s != self.target:
                continue

            if path.exists(angels_full_path):
                self.data[s]['angels pdb'] = pkl_load(angels_full_path)
                self.data[s]['angels pdb'][self.data[s]['angels pdb'] ==
                                           -1] = np.nan
            else:
                logging.info('finding angels for %s' % s)
                self.find_aligned_angels(uniprot=s,
                                         sifts_mapping=create_sifts_mapping())

    def find_pdb_cover(self, uniprot, sifts_mapping):
        if uniprot == self.target:
            pdb_start = 0
            pdb_end = len(list(self.protein.sequence))
            return {uniprot: (pdb_start, pdb_end)}
        elif uniprot not in sifts_mapping:
            return

        selected_structures = sifts_mapping[uniprot]
        return {
            p[0][0:5]: (p[1]['pdb'][0] - 1, p[1]['pdb'][1] - 1)
            for p in selected_structures
        }

    def align_pdb_cover(self, cover, pdb_start, pdb_end, seq_msa):
        """This function takes cover name and sifts indices
        and returns the pdb and msa covered indices if they exist"""

        seq_msa_no_gaps = ''.join([aa for aa in seq_msa if aa != '-']).upper()
        protein = cover[0:4]
        chain = cover[4]
        try:
            reference_protein = Protein(protein, chain)
            seq_pdb = ''.join(reference_protein.sequence)
            seq_pdb_short = seq_pdb[pdb_start:pdb_end]
        except (FileNotFoundError, ValueError, IndexError) as e:
            return
        full_pdb_l = len(seq_pdb)
        short_pdb_l = len(seq_pdb_short)
        # we want at least 50 % cover
        insufficient_cover = pdb_end - pdb_start < full_pdb_l / 2 or pdb_end > full_pdb_l
        if insufficient_cover:
            return

        msa_in_pdb_start = seq_pdb.find(seq_msa_no_gaps)
        pdb_full_in_msa_start = seq_msa_no_gaps.find(seq_pdb)
        pdb_sifts_in_msa_start = seq_msa_no_gaps.find(seq_pdb_short)

        is_msa_in_pdb = msa_in_pdb_start != -1
        is_pdb_full_in_msa = pdb_full_in_msa_start != -1
        is_pdb_sifts_in_msa = pdb_sifts_in_msa_start != -1

        unmatching_sequences = not is_msa_in_pdb and not is_pdb_full_in_msa and not is_pdb_sifts_in_msa

        if unmatching_sequences:
            return
        list_seq_enumeration = list(enumerate(seq_msa))
        msa_indices = [aa[0] for aa in list_seq_enumeration if aa[1] != '-']
        if is_msa_in_pdb:
            pdb_start = msa_in_pdb_start
            pdb_end = msa_in_pdb_start + len(seq_msa_no_gaps)
            msa_no_gaps_start = 0
            msa_no_gaps_end = len(seq_msa_no_gaps)
        elif is_pdb_full_in_msa:
            pdb_start = 0
            pdb_end = full_pdb_l
            msa_no_gaps_start = pdb_full_in_msa_start
            msa_no_gaps_end = pdb_full_in_msa_start + full_pdb_l
        elif is_pdb_sifts_in_msa:
            msa_no_gaps_start = pdb_sifts_in_msa_start
            msa_no_gaps_end = pdb_sifts_in_msa_start + short_pdb_l
        else:
            return
        msa_indices_covered = msa_indices[msa_no_gaps_start:msa_no_gaps_end]
        pdb_indices = list(range(pdb_start, pdb_end))

        if not len(pdb_indices) == len(msa_indices_covered):
            logging.critical('could not align %s to %s' % (cover, self.target))
            return

        return {'msa': msa_indices_covered, 'pdb': pdb_indices}

    @timefunc
    def find_cover(self, uniprot, sifts_mapping):

        pdb_covers = self.find_pdb_cover(uniprot=uniprot,
                                         sifts_mapping=sifts_mapping)
        if not pdb_covers:
            return

        seq_msa = list(self.data[uniprot]['seq'])
        for cover, (pdb_start, pdb_end) in pdb_covers.items():
            aligned_cover = self.align_pdb_cover(cover=cover,
                                                 pdb_start=pdb_start,
                                                 pdb_end=pdb_end,
                                                 seq_msa=seq_msa)
            if aligned_cover:
                return {'name': cover, 'pdb_msa_map': aligned_cover}

    def find_aligned_distmat(self, uniprot, sifts_mapping):
        aligned_cover = self.find_cover(uniprot, sifts_mapping)
        seq_msa = list(self.data[uniprot]['seq'])

        aligned_dist_mat = np.empty(shape=(len(seq_msa), len(seq_msa)))
        aligned_dist_mat[:] = np.nan

        pdb_cols = aligned_cover['pdb_msa_map']['pdb']
        pdb_rows = [[aa] for aa in pdb_cols]

        msa_cols = aligned_cover['pdb_msa_map']['msa']
        msa_rows = [[aa] for aa in msa_cols]

        cover_protein = aligned_cover['name'][0:4]
        cover_chain = aligned_cover['name'][4]
        pdb_dm = Protein(cover_protein, cover_chain).dm

        aligned_dist_mat[msa_rows, msa_cols] = pdb_dm[pdb_rows, pdb_cols]
        # aligned_dist_mat += aligned_dist_mat.transpose()

        dm_msa = pd.DataFrame(aligned_dist_mat, columns=seq_msa, index=seq_msa)
        self.data[uniprot]['dist mat pdb'] = self.msa_mat_to_pdb(dm_msa)
        # print(protein + chain)
        self.known_structures.add(uniprot)
        self.data[uniprot]['cover'] = aligned_cover['name']
        structure_file = path.join(self.msa_data_path, '%s_pdb.pkl' % uniprot)

        pkl_save(structure_file, self.data[uniprot]['dist mat pdb'])

    @timefunc
    def find_aligned_angels(self, uniprot, sifts_mapping):
        aligned_cover = self.find_cover(uniprot, sifts_mapping)
        seq_msa = list(self.data[uniprot]['seq'])

        aligned_angels = np.empty(shape=(len(seq_msa), 2))
        aligned_angels[:] = np.nan

        pdb_inds = aligned_cover['pdb_msa_map']['pdb']

        msa_inds = aligned_cover['pdb_msa_map']['msa']

        cover_protein = aligned_cover['name'][0:4]
        cover_chain = aligned_cover['name'][4]
        pdb_angels = Protein(cover_protein, cover_chain).angels
        aligned_angels[msa_inds, :] = pdb_angels[pdb_inds, :]

        self.data[uniprot]['angels pdb'] = self.angels_arr_to_pdb(
            aligned_angels)

        self.angels_file = path.join(self.msa_data_path,
                                     '%s_angels.pkl' % uniprot)

        pkl_save(self.angels_file, self.data[uniprot]['angels pdb'])

    def find_structure(self, uniprot, sifts_mapping, pdb=False):
        '''
        :param uniprot: homologos uniprot id
        :return: aligned distmat
        '''
        if pdb:
            protein = uniprot[:4]
            chain = uniprot[4]
            seq = list(self.protein.sequence)

            start_pdb = 0
            end_pdb = len(seq)
            dist_mat = get_dist_mat(protein, chain).iloc[start_pdb:end_pdb,
                                                         start_pdb:end_pdb]
            seq_pdb = ''.join(list(dist_mat.columns))
            list_seq_msa = list(self.data[uniprot]['seq'])
            seq_msa = ''.join(list_seq_msa)
            seq_msa_no_gaps = seq_msa.replace("-", "").upper()
            subset_ind = seq_pdb.find(seq_msa_no_gaps)
            subset_ind_2 = seq_msa_no_gaps.find(seq_pdb)
            msa_in_pdb = subset_ind != -1
            pdb_in_msa = subset_ind_2 != -1

            if msa_in_pdb or pdb_in_msa:
                aligned_dist_mat = np.empty(shape=(len(seq_msa), len(seq_msa)))
                aligned_dist_mat[:] = np.nan
                list_seq_enumeration = list(enumerate(list_seq_msa))
                distances_indices = [
                    aa[0] for aa in list_seq_enumeration if aa[1] != '-'
                ]
                distances_indices_rows = [[aa] for aa in distances_indices]
                if msa_in_pdb:
                    dist_mat = dist_mat.iloc[subset_ind:subset_ind +
                                             len(seq_msa_no_gaps),
                                             subset_ind:subset_ind +
                                             len(seq_msa_no_gaps)]
                    aligned_dist_mat[distances_indices_rows,
                                     distances_indices] = np.array(dist_mat)

                elif pdb_in_msa:
                    start_msa = subset_ind_2
                    end_msa = subset_ind_2 + len(seq_pdb)
                    aligned_dist_mat[
                        distances_indices_rows[start_msa:end_msa],
                        distances_indices[start_msa:end_msa]] = np.array(
                            dist_mat)
                dm_msa = pd.DataFrame(aligned_dist_mat,
                                      columns=list_seq_msa,
                                      index=list_seq_msa)
                self.data[uniprot]['dist mat pdb'] = self.msa_mat_to_pdb(
                    dm_msa)
                self.known_structures.add(protein + chain)
                # self.data[uniprot]['cover'] = protein + chain
                self.metadata['structures'][uniprot] = protein + chain
                structure_file = path.join(self.msa_data_path,
                                           '%s_pdb.pkl' % uniprot)

                pkl_save(structure_file, self.data[uniprot]['dist mat pdb'])

        else:
            if uniprot not in sifts_mapping:
                return
            selected_structures = sifts_mapping[uniprot]

            list_seq_msa = list(self.data[uniprot]['seq'])
            seq_msa = ''.join(list_seq_msa).upper()
            seq_msa_no_gaps = seq_msa.replace("-", "")

            list_seq_msa_target = list(self.data[self.target]['seq'])
            seq_msa_target = ''.join(list_seq_msa_target).upper()
            seq_msa_target_no_gaps = seq_msa_target.replace('-', '')
            seq_mas_target_no_gaps_length = len(seq_msa_target_no_gaps)

            for protein_map in selected_structures:
                protein = protein_map[0][0:4]
                chain = protein_map[0][4]
                if protein == self.target[0:4]:
                    continue
                try:
                    reference_protein = Protein(protein, chain)
                except (FileNotFoundError, IndexError) as e:
                    continue
                start_pdb, end_pdb = protein_map[1]['pdb'][0] - 1, protein_map[
                    1]['pdb'][1] - 1

                try:
                    seq_pdb = ''.join(reference_protein.sequence)

                    seq_pdb_short = seq_pdb[start_pdb:end_pdb]
                except ValueError:
                    continue

                full_pdb_l = len(seq_pdb)
                short_pdb_l = len(seq_pdb_short)

                if end_pdb - start_pdb < full_pdb_l / 2 or end_pdb > full_pdb_l:
                    continue
                subset_ind = seq_pdb.find(seq_msa_no_gaps)
                subset_ind_2 = seq_msa_no_gaps.find(seq_pdb)
                subset_ind_3 = seq_msa_no_gaps.find(seq_pdb_short)

                msa_in_pdb = subset_ind != -1
                pdb_in_msa = subset_ind_2 != -1
                pdb_short_in_msa = subset_ind_3 != -1

                if msa_in_pdb or pdb_in_msa or pdb_short_in_msa:
                    dist_mat = get_dist_mat(protein, chain)
                    if pdb_short_in_msa and not pdb_in_msa and not msa_in_pdb:
                        seq_pdb = seq_pdb_short
                        dist_mat = dist_mat.iloc[start_pdb:end_pdb,
                                                 start_pdb:end_pdb]
                        subset_ind_2 = subset_ind_3
                    aligned_dist_mat = np.empty(shape=(len(seq_msa),
                                                       len(seq_msa)))
                    aligned_dist_mat[:] = np.nan
                    list_seq_enumeration = list(enumerate(list_seq_msa))
                    distances_indices = [
                        aa[0] for aa in list_seq_enumeration if aa[1] != '-'
                    ]
                    distances_indices_rows = [[aa] for aa in distances_indices]
                    if msa_in_pdb:
                        dist_mat = dist_mat.iloc[subset_ind:subset_ind +
                                                 len(seq_msa_no_gaps),
                                                 subset_ind:subset_ind +
                                                 len(seq_msa_no_gaps)]
                        coverage_identity = 0
                        for i in range(len(seq_msa_target)):
                            if seq_msa_target[i] == seq_msa[
                                    i] and seq_msa[i] != '-':
                                coverage_identity += 1
                        # print('found coverage')

                        self.data[uniprot][
                            'coverage_identity'] = coverage_identity / seq_mas_target_no_gaps_length

                        aligned_dist_mat[distances_indices_rows,
                                         distances_indices] = np.array(
                                             dist_mat)

                    elif pdb_in_msa or pdb_short_in_msa:
                        start_msa = subset_ind_2
                        end_msa = subset_ind_2 + len(seq_pdb)
                        aligned_dist_mat[
                            distances_indices_rows[start_msa:end_msa],
                            distances_indices[start_msa:end_msa]] = np.array(
                                dist_mat)
                        coverage_identity = 0
                        sub_ind = 0
                        for i in range(len(seq_msa_target)):
                            if seq_msa_target[i] == seq_msa[i] and seq_msa[
                                    i] != '-' and sub_ind >= subset_ind_2:
                                coverage_identity += 1
                            if seq_msa[i] != '-':
                                sub_ind += 1
                        # print('found coverage')
                        self.data[uniprot][
                            'coverage_identity'] = coverage_identity / seq_mas_target_no_gaps_length

                    dm_msa = pd.DataFrame(aligned_dist_mat,
                                          columns=list_seq_msa,
                                          index=list_seq_msa)
                    self.data[uniprot]['dist mat pdb'] = self.msa_mat_to_pdb(
                        dm_msa)
                    # print(protein + chain)
                    self.known_structures.add(uniprot)
                    # self.data[uniprot]['cover'] = protein + chain
                    self.metadata['structures'][uniprot] = protein + chain
                    structure_file = path.join(self.msa_data_path,
                                               '%s_pdb.pkl' % uniprot)

                    pkl_save(structure_file,
                             self.data[uniprot]['dist mat pdb'])
                return None

    @timefunc
    def find_all_structures(self):
        logging.info('finding structures for msa %s' % self.target)
        self.metadata['structures'] = {}
        self.known_structures = set()

        sifts_mapping = create_sifts_mapping()
        # if self.metadata:
        #     self.load_known_structures()
        # elif not train:
        self.data = read_fasta(self.msa_filename, True)
        self.find_structure(self.target, sifts_mapping, pdb=True)
        mas_homologous = set(list(self.data.keys())).difference(self.target)
        for homologous in mas_homologous:
            try:
                self.find_structure(homologous, sifts_mapping)
            except ValueError:
                continue
        for hom in self.data:
            if self.data[hom]['dist mat pdb'] is not None:
                self.known_structures.add(hom)
        self.metadata['structures_date'] = datetime.date.today().strftime(
            "%B %d, %Y")
        pkl_save(data=self.metadata, filename=self.metadata_fname)

    def na_matrix(self, matrix):
        matrix[matrix == -1] = np.nan
        matrix_dist = np.array(matrix)
        matrix_dist[np.isnan(np.array(matrix, dtype=float))] = 0
        matrix_na = np.zeros_like(matrix)
        matrix_na[np.isnan(np.array(matrix, dtype=float))] = 1
        new_shape = (matrix.shape[0], matrix.shape[1], 1)
        return np.concatenate(
            (matrix_dist.reshape(new_shape), matrix_na.reshape(new_shape)),
            axis=2)

    def msa_mat_to_pdb(self, msa_mat):
        if msa_mat.shape != (self.target_seq_length_msa,
                             self.target_seq_length_msa):
            raise IndexError('msa mat is not valid')
        pdb_mat = np.empty(shape=(self.target_seq_length_pdb,
                                  self.target_seq_length_pdb))
        pdb_mat[:] = np.nan

        target_inds_msa = self.target_indices_msa
        target_inds_msa_transpose = (self.target_indices_msa[1],
                                     self.target_indices_msa[0])

        target_inds_pdb = self.target_indices_pdb
        target_inds_pdb_transpose = (self.target_indices_pdb[1],
                                     self.target_indices_pdb[0])

        if type(msa_mat) == pd.core.frame.DataFrame:
            pdb_mat[target_inds_pdb] = msa_mat.values[target_inds_msa]
            pdb_mat[target_inds_pdb_transpose] = msa_mat.values[
                target_inds_msa_transpose]

        else:
            pdb_mat[self.target_indices_pdb] = msa_mat[self.target_indices_msa]

        return pdb_mat

    def angels_arr_to_pdb(self, angels_array, na_value=-1):

        if angels_array.shape != (self.target_seq_length_msa, 2):
            raise IndexError('msa mat is not valid')

        pdb_angels_array = np.ones(shape=(self.target_seq_length_pdb,
                                          2)) * na_value
        pdb_angels_array[self.target_indices_pdb[0], :] = angels_array[
            self.target_indices_msa[0], :]

        return pdb_angels_array

    @property
    def mean_known_structures_dm(self):
        mean_distance_final = self.na_matrix(self.mean_known_structures_dm_na)
        return mean_distance_final

    @property
    def mean_known_structures_dm_na(self):
        structure_matrices = []
        for struc in self.known_structures:
            if struc != self.target:
                dm = self.data[struc]['dist mat pdb']
                dm[dm == -1] = np.nan
                structure_matrices.append(dm)
        mean_distance = np.nanmean(structure_matrices, axis=0)
        return mean_distance

    @property
    def closest_known_structure_angels(self):
        angels = self.data[self.closest_known_strucutre]['angels pdb']
        na_angels = np.zeros_like(angels)
        na_angels[np.isnan(angels)] = 1
        angels[np.isnan(angels)] = 0
        angels_full = np.concatenate([angels, na_angels], axis=1)
        return angels_full

    @property
    def closest_known_structure_dm(self):
        dm = self.na_matrix(self.closest_known_structure_dm_na)
        return dm

    @property
    def closest_known_structure_dm_na(self):
        struc = self.closest_known_strucutre
        dm = self.data[struc]['dist mat pdb']
        dm[dm == -1] = np.nan
        return dm

    @timefunc
    def run_plmc(self, force, version):
        dir_msa = '/cs/zbio/orzuk/projects/ContactMaps/data/MSA_Completion/msa_data'
        file_ec = '%s/dca/%s_v%s.txt' % (dir_msa, self.target, version)
        if not path.isfile(file_ec) or force:
            file_params = '%s/dca/%s_%s.params' % (dir_msa, self.target,
                                                   version)
            file_msa = '%s/hhblits/fasta/%s_v%s.fasta' % (dir_msa, self.target,
                                                          version)
            cmd = 'plmc -o %s -c %s -le 16.0 -m %s -lh 0.01  -g -f %s %s' \
                  % (file_params, file_ec, 500, self.target, file_msa)
            subprocess.call(cmd)

    def get_ec(self, version):
        dir_msa = '/cs/zbio/orzuk/projects/ContactMaps/data/MSA_Completion/msa_data'
        file_ec = '%s/dca/%s_v2.txt' % (dir_msa, self.target)
        if not path.isfile(file_ec):
            self.run_plmc(version=version, force=False)
        ec = read_raw_ec_file(file_ec)

        list_seq_msa = list(self.data[self.target]['seq'])
        seq_msa = ''.join(list_seq_msa)
        seq_msa_no_gaps = seq_msa.replace("-", "").upper()
        seq_pdb = ''.join(self.target_seq)
        subset_ind = seq_pdb.find(seq_msa_no_gaps)
        subset_ind_2 = seq_msa_no_gaps.find(seq_pdb)
        if subset_ind == -1 and subset_ind_2 == -1:
            raise IndexError('plmc does not match msa indices')
        pdb_msa_map = {}
        index_pdb = max(subset_ind, 0)
        index_msa = max(subset_ind_2, 0)

        n_gaps = 0
        for msa_ind in range(max(subset_ind_2, 0), len(seq_msa)):
            if seq_msa[msa_ind] != '-':
                if index_pdb < len(seq_pdb):
                    pdb_msa_map[index_msa + n_gaps + 1] = index_pdb
                else:
                    pdb_msa_map[index_msa + n_gaps + 1] = None

                index_msa += 1
                index_pdb += 1
            else:
                n_gaps += 1

        ec['pdb_i'] = ec['i'].map(pdb_msa_map)
        ec['pdb_j'] = ec['j'].map(pdb_msa_map)
        a = []
        for k, v in pdb_msa_map.items():
            if v is None:
                continue

            a.append(seq_msa[k - 1].upper() == seq_pdb[v])
        if np.mean(a) != 1:
            raise IndexError('msa and pdb sequences dont match')
        return ec

    @timefunc
    def read_plmc_data(self, version):
        ec = self.get_ec(version=version)

        self.target_indices_msa = (np.array(ec['i'] - 1).astype(np.int64),
                                   np.array(ec['j'] - 1).astype(np.int64))
        self.target_indices_pdb = (np.array(ec['pdb_i']).astype(np.int64),
                                   np.array(ec['pdb_j']).astype(np.int64))

        self.plmc_score = np.zeros(shape=(self.target_seq_length_pdb,
                                          self.target_seq_length_pdb))

        self.plmc_score[self.target_indices_pdb] = ec['cn'].values

    @timefunc
    def structures_dist_mat(self):
        if len(self.known_structures) == 0:
            return None
        phylo_structures_file = path.join(self.msa_data_path,
                                          'structures_phylo_dist_mat_v2.pkl')
        if path.isfile(phylo_structures_file):
            phylo_structures_mat = pd.read_pickle(phylo_structures_file)
            if set(phylo_structures_mat.index) == self.known_structures:
                self.phylo_structures_mat = phylo_structures_mat
                return None

        structures_list = list(self.known_structures)
        msa_structures = [
            str(self.data[s]['seq']).upper() for s in structures_list
        ]
        id_mat = compute_structures_identity_matrix(msa_structures,
                                                    msa_structures,
                                                    target=self.target)
        identity_mat = pd.DataFrame(id_mat,
                                    columns=structures_list,
                                    index=structures_list)
        self.phylo_structures_mat = 1 - identity_mat

        self.phylo_structures_mat.to_pickle(phylo_structures_file)

    @property
    def closest_known_strucutre(self, far=False):

        struc_dm = self.phylo_structures_mat
        if far:
            structure = struc_dm.loc[:, self.target].sort_values().index[-1]
        else:
            structure = struc_dm.loc[:, self.target].sort_values().index[1]
        if self.data[structure]['dist mat pdb'] is None:
            raise ValueError('No dist mat for structure')

        return structure

    @property
    def aa_numeric_dict(self):
        aa = list('-ACDEFGHIKLMNPQRSTVWYX')
        aa_numeirc_dict = {}
        power = -3
        i = 1
        nums = [1, 3, 4]
        ind = 0
        for aa_1 in aa:
            aa_numeirc_dict[aa_1] = (i * 10.0)**power

            if power == 0:
                power += 1
            if ind == 2:
                power += 1
                ind = 0
            else:
                ind += 1
            i = nums[ind]
        aa_numeirc_dict['Z'] = aa_numeirc_dict['-']
        aa_numeirc_dict['B'] = aa_numeirc_dict['-']
        aa_numeirc_dict['U'] = aa_numeirc_dict['-']
        aa_numeirc_dict['O'] = aa_numeirc_dict['-']
        return aa_numeirc_dict

    @property
    def pairs_dict(self):
        aa = list('-ACDEFGHIKLMNPQRSTVWYX')
        aa_pairs_dict = {}
        for a in aa:
            for b in aa:
                aa_pairs_dict[
                    a + b] = self.aa_numeric_dict[a] + self.aa_numeric_dict[b]
        a = pd.DataFrame({
            'pair': list(aa_pairs_dict.keys()),
            'n': list(aa_pairs_dict.values())
        })
        a = a.sort_values('n')
        return a

    @property
    def aa_dict(self):
        aa = list('-ACDEFGHIKLMNPQRSTVWYX')
        aa_dict = {aa[i]: i for i in range(len(aa))}
        aa_dict['Z'] = 0
        aa_dict['B'] = 0
        aa_dict['U'] = 0
        aa_dict['O'] = 0
        return aa_dict

    @property
    def modeller_dm(self):
        modeller_path = '/cs/zbio/orzuk/projects/ContactMaps/data/MSA-Completion/models/Modeller_data'
        target_path = path.join(modeller_path, self.target)

        target_pdb_fname = 'v%s_pdb' % self.version + self.protein_name + '.ent'

        target_modeller_pdb_file = path.join(target_path, target_pdb_fname)
        if not path.isfile(target_modeller_pdb_file):
            reference = self.closest_known_strucutre
            closest_structure = self.metadata['structures'][reference]
            template = closest_structure[0:4]
            template_chain = closest_structure[4]
            mod_args = (self.protein_name, self.chain_name, template,
                        template_chain)
            cmd = '/cs/staff/dina/modeller9.18/bin/modpy.sh python3 run_modeller.py %s %s %s %s' % mod_args
            logging.info('executing %s' % cmd)
            try:
                subprocess.call(cmd)
            except urllib.error.URLError:
                raise FileNotFoundError('pdb download error')

            mod_args_2 = (self.protein_name, self.chain_name, template,
                          template_chain, self.version)

            cmd2 = '/cs/staff/dina/modeller9.18/bin/modpy.sh python3 modeller_files.py %s %s %s %s %s' % mod_args_2
            logging.info('executing %s' % cmd2)
            subprocess.call(cmd2)
        dm_modeller_initial = modeller_calc_dist_matrix(
            self.protein_name, self.chain_name, self.version)

        seq_modeller = "".join(dm_modeller_initial.index)

        bad_cols = []
        j = 0

        for i in range(len(self.target_seq)):
            if j > len(seq_modeller) - 1:
                bad_cols.append(i)
            elif self.target_seq[i] != seq_modeller[j]:
                bad_cols.append(i)
            else:
                j += 1

        bad_cols = np.array(bad_cols)
        pairs = [(i, j) for i, j in itertools.combinations(
            range(self.target_seq_length_pdb), 2)
                 if i < j and i not in bad_cols and j not in bad_cols]
        pairs_modeller = [(pair[0] - np.sum(pair[0] >= bad_cols),
                           pair[1] - np.sum(pair[1] >= bad_cols))
                          for pair in pairs]

        rows, cols = zip(*pairs)
        rows_modeller, cols_modeller = zip(*pairs_modeller)
        modeller_dm = np.ones_like(self.target_pdb_cm) * np.nan

        modeller_dm[rows, cols] = dm_modeller_initial.values[rows_modeller,
                                                             cols_modeller]

        return modeller_dm

    def get_stability_scores_2(self):
        homologos = list(self.data.keys())
        self.evolutinairy_scores()
        lost_pairs_log_ratio_full = np.zeros(
            shape=(self.target_seq_length_pdb, self.target_seq_length_pdb))
        preserved_pairs_log_ratio_full = np.zeros_like(
            lost_pairs_log_ratio_full)
        created_pairs_log_ratio_full = np.zeros_like(lost_pairs_log_ratio_full)

        alignment = self.get_sub_alignment(homologos)
        msa_inds = zip(self.target_indices_msa[0], self.target_indices_msa[1])
        pdb_inds = list(
            zip(self.target_indices_pdb[0], self.target_indices_pdb[1]))
        count = 0
        for i, j in msa_inds:

            pdb_i, pdb_j = pdb_inds[count]
            pair = np.core.defchararray.add(alignment[:, i], alignment[:, j])
            vals, counts = np.unique(pair, return_counts=True)
            for v, c in zip(vals, counts):
                lost_pairs_log_ratio_full[
                    pdb_i, pdb_j] += c * self.lost_log_ratio_dict[v]
                lost_pairs_log_ratio_full[
                    pdb_j, pdb_i] += c * self.lost_log_ratio_dict[v]

                preserved_pairs_log_ratio_full[
                    pdb_i, pdb_j] += c * self.preserved_log_ratio_dict[v]
                preserved_pairs_log_ratio_full[
                    pdb_j, pdb_i] += c * self.preserved_log_ratio_dict[v]

                created_pairs_log_ratio_full[
                    pdb_i, pdb_j] += c * self.created_log_ratio_dict[v]
                created_pairs_log_ratio_full[
                    pdb_j, pdb_i] += c * self.created_log_ratio_dict[v]
            count += 1
        log_ratio_arrays = (np.expand_dims(lost_pairs_log_ratio_full, axis=2),
                            np.expand_dims(created_pairs_log_ratio_full,
                                           axis=2),
                            np.expand_dims(preserved_pairs_log_ratio_full,
                                           axis=2))
        return log_ratio_arrays

    def evolutinairy_scores(self):
        uni_freq = uniprot_frequency()
        evo_freq = evolutionairy_frequency()
        lost_log_ratio = np.maximum(np.log(evo_freq['lost'] / uni_freq), -10)
        created_log_ratio = np.maximum(np.log(evo_freq['gain'] / uni_freq),
                                       -10)
        preserved_log_ratio = np.maximum(
            np.log(evo_freq['maintained'] / uni_freq), -10)
        aas = list(uni_freq.index)
        lost_log_ratio_dict = {}
        created_log_ratio_dict = {}
        preserved_log_ratio_dict = {}
        lost_log_ratio_dict_numeric = {}
        created_log_ratio_dict_numeric = {}
        preserved_log_ratio_dict_numeric = {}
        aa_numeric = self.aa_numeric_dict
        for aa_1 in aas:
            for aa_2 in aas:
                lost_log_ratio_dict[aa_1 + aa_2] = lost_log_ratio.loc[aa_1,
                                                                      aa_2]
                lost_log_ratio_dict_numeric[
                    aa_numeric[aa_1] +
                    aa_numeric[aa_2]] = lost_log_ratio.loc[aa_1, aa_2]

                created_log_ratio_dict[aa_1 +
                                       aa_2] = created_log_ratio.loc[aa_1,
                                                                     aa_2]
                created_log_ratio_dict_numeric[
                    aa_numeric[aa_1] +
                    aa_numeric[aa_2]] = created_log_ratio.loc[aa_1, aa_2]

                preserved_log_ratio_dict[aa_1 +
                                         aa_2] = preserved_log_ratio.loc[aa_1,
                                                                         aa_2]
                preserved_log_ratio_dict_numeric[
                    aa_numeric[aa_1] +
                    aa_numeric[aa_2]] = preserved_log_ratio.loc[aa_1, aa_2]

        gap = '-'
        lost_log_ratio_dict[gap + gap] = 0
        lost_log_ratio_dict_numeric[aa_numeric[gap] * 2] = 0
        created_log_ratio_dict[gap + gap] = 0
        created_log_ratio_dict_numeric[aa_numeric[gap] * 2] = 0
        preserved_log_ratio_dict[gap + gap] = 0
        preserved_log_ratio_dict_numeric[aa_numeric[gap] * 2] = 0

        for aa in aas:
            lost_log_ratio_dict[aa + gap] = np.sum(
                lost_log_ratio.loc[aa, :].values * self.aa_freq_df.values)
            lost_log_ratio_dict[gap + aa] = lost_log_ratio_dict[aa + gap]
            lost_log_ratio_dict_numeric[
                aa_numeric[gap] + aa_numeric[aa]] = lost_log_ratio_dict[aa +
                                                                        gap]
            lost_log_ratio_dict_numeric[
                aa_numeric[aa] + aa_numeric[gap]] = lost_log_ratio_dict[aa +
                                                                        gap]

            created_log_ratio_dict[aa + gap] = np.sum(
                created_log_ratio.loc[aa, :].values * self.aa_freq_df.values)
            created_log_ratio_dict[gap + aa] = created_log_ratio_dict[aa + gap]
            created_log_ratio_dict_numeric[
                aa_numeric[gap] + aa_numeric[aa]] = created_log_ratio_dict[aa +
                                                                           gap]
            created_log_ratio_dict_numeric[
                aa_numeric[aa] + aa_numeric[gap]] = created_log_ratio_dict[aa +
                                                                           gap]

            preserved_log_ratio_dict[aa + gap] = np.sum(
                preserved_log_ratio.loc[aa, :].values * self.aa_freq_df.values)
            preserved_log_ratio_dict[gap + aa] = preserved_log_ratio_dict[aa +
                                                                          gap]
            preserved_log_ratio_dict_numeric[
                aa_numeric[gap] +
                aa_numeric[aa]] = preserved_log_ratio_dict[aa + gap]
            preserved_log_ratio_dict_numeric[
                aa_numeric[aa] +
                aa_numeric[gap]] = preserved_log_ratio_dict[aa + gap]

        self.preserved_log_ratio_dict = preserved_log_ratio_dict
        self.created_log_ratio_dict = created_log_ratio_dict
        self.lost_log_ratio_dict = lost_log_ratio_dict
        self.preserved_log_ratio_dict_numeric = preserved_log_ratio_dict_numeric
        self.lost_log_ratio_dict_numeric = lost_log_ratio_dict_numeric
        self.created_log_ratio_dict_numeric = created_log_ratio_dict_numeric

    def get_sub_alignment(self, homologos):
        seqs = [
            SeqRecord(self.data[entry.split('_')[0]]['seq'].upper(),
                      id=entry.split('_')[0]) for entry in homologos
        ]

        return np.array(seqs)

    @property
    def sequence_distance_matrix(self):
        def i_minus_j(i, j):
            return np.abs(i - j)

        sequence_distance = np.fromfunction(i_minus_j,
                                            shape=self.target_pdb_cm.shape)
        return sequence_distance

    # @property
    # def stability_scores(self):
    #     if 'stability_scores' not in self._features.keys():
    #         self._features['stability_scores'] = self._get_stability_scores()
    #     return self._features['stability_scores']

    # def _get_dict_map(self, map_dict):
    #     return njit_index_dict(np.array(list(map_dict.keys()), np.float64),
    #                            np.array(list(map_dict.values()), np.float64))

    # def _get_stability_scores(self):
    #     homologos = list(self.data.keys())
    #     self.evolutinairy_scores()
    #
    #     aa_numeric_dict = self.aa_numeric_dict
    #     alignment = self.get_sub_alignment(homologos)
    #     alignment_numeric_lr = np.vectorize(aa_numeric_dict.__getitem__)(alignment)
    #     msa_indices = list(zip(self.target_indices_msa[0], self.target_indices_msa[1]))
    #
    #     pairs_alignment_full = slice_sum_array(alignment_numeric_lr, msa_indices)
    #     shp = (self.target_seq_length_pdb, self.target_seq_length_pdb)
    #
    #     dict_map = self._get_dict_map(self.lost_log_ratio_dict_numeric)
    #     lost_pairs_log_ratio_full = np.zeros(shape=shp)
    #     lost_pairs_log_ratio_full[self.target_indices_pdb] = stability_batchinng(alignment=pairs_alignment_full,
    #                                                                              dict_map=dict_map,
    #                                                                              seq_length=self.target_seq_length_pdb)
    #     lost_pairs_log_ratio_full += lost_pairs_log_ratio_full.transpose()
    #
    #     dict_map = self._get_dict_map(self.created_log_ratio_dict_numeric)
    #     created_pairs_log_ratio_full = np.zeros(shape=shp)
    #     created_pairs_log_ratio_full[self.target_indices_pdb] = stability_batchinng(alignment=pairs_alignment_full,
    #                                                                                 dict_map=dict_map,
    #                                                                                 seq_length=self.target_seq_length_pdb)
    #     created_pairs_log_ratio_full += created_pairs_log_ratio_full.transpose()
    #
    #     dict_map = self._get_dict_map(self.preserved_log_ratio_dict_numeric)
    #     preserved_pairs_log_ratio_full = np.zeros(shape=shp)
    #     preserved_pairs_log_ratio_full[self.target_indices_pdb] = stability_batchinng(alignment=pairs_alignment_full,
    #                                                                                   dict_map=dict_map,
    #                                                                                   seq_length=self.target_seq_length_pdb)
    #     preserved_pairs_log_ratio_full += preserved_pairs_log_ratio_full.transpose()
    #
    #     return np.concatenate((np.expand_dims(lost_pairs_log_ratio_full, axis=2),
    #                            np.expand_dims(created_pairs_log_ratio_full, axis=2),
    #                            np.expand_dims(preserved_pairs_log_ratio_full, axis=2)), axis=2)

    # def get_stability_scores_old(self):
    #     homologos = list(self.data.keys())
    #     self.evolutinairy_scores()
    #
    #     aa_numeric_dict = self.aa_numeric_dict
    #     alignment = self.get_sub_alignment(homologos)
    #     alignment_numeric_lr = np.vectorize(aa_numeric_dict.__getitem__)(alignment)
    #     msa_indices = list(zip(self.target_indices_msa[0], self.target_indices_msa[1]))
    #
    #     pairs_alignment_full = slice_sum_array(alignment_numeric_lr, msa_indices)
    #
    #     batch_size = 200
    #     homs = alignment.shape[0]
    #     batches = np.ceil(homs / batch_size)
    #     lost_pairs_log_ratio_full = np.zeros(shape=(self.target_seq_length_pdb, self.target_seq_length_pdb))
    #     preserved_pairs_log_ratio_full = np.zeros_like(lost_pairs_log_ratio_full)
    #     created_pairs_log_ratio_full = np.zeros_like(lost_pairs_log_ratio_full)
    #
    #     for b in range(int(batches)):
    #         rows = range(b * batch_size, min(homs, (b + 1) * batch_size))
    #         pairs_alignment = pairs_alignment_full[rows]
    #
    #         lost_pairs_log_ratio_full[self.target_indices_pdb] += replace_with_dict_wrapper(pairs_alignment,
    #                                                                                         aa_numeric_dict,
    #                                                                                         self.lost_log_ratio_dict)[0,
    #                                                               :]
    #         preserved_pairs_log_ratio_full[self.target_indices_pdb] += replace_with_dict_wrapper(pairs_alignment,
    #                                                                                              aa_numeric_dict,
    #                                                                                              self.preserved_log_ratio_dict)[
    #                                                                    0, :]
    #         created_pairs_log_ratio_full[self.target_indices_pdb] += replace_with_dict_wrapper(pairs_alignment,
    #                                                                                            aa_numeric_dict,
    #                                                                                            self.created_log_ratio_dict)[
    #                                                                  0, :]
    #
    #     log_ratio_arrays = (np.expand_dims(lost_pairs_log_ratio_full, axis=2),
    #                         np.expand_dims(created_pairs_log_ratio_full, axis=2),
    #                         np.expand_dims(preserved_pairs_log_ratio_full, axis=2))
    #     log_ratio_concat = np.concatenate(log_ratio_arrays, axis=2)
    #     return log_ratio_concat

    def get_pssm(self, homologos):

        alignment = self.get_sub_alignment(homologos)
        alignment_pdb = np.zeros(shape=(len(homologos),
                                        self.target_seq_length_pdb),
                                 dtype='<U1')
        alignment_pdb[:] = '-'
        alignment_pdb[:, self.
                      target_indices_pdb[0]] = alignment[:, self.
                                                         target_indices_msa[0]]
        numeric_msa = np.vectorize(
            self.aa_dict.__getitem__)(alignment_pdb).astype(np.int)
        pssm = compute_pssm(numeric_msa)
        pkl_save(path.join(self.msa_data_path, 'pssm.pkl'), pssm)
        return pssm

    @property
    def pssm(self):
        # shape l*l*44
        pssm_file = path.join(self.msa_data_path, 'pssm.pkl')
        if path.isfile(pssm_file):
            return pkl_load(pssm_file)
        else:
            return self.get_pssm(homologos=list(self.data.keys()))

    @property
    def outer_angels(self):
        # shape l*l*4
        angels = self.protein.angels
        angels[np.isnan(angels)] = 0

        angels00 = np.outer(angels[:, 0], angels[:, 0])
        angels01 = np.outer(angels[:, 0], angels[:, 1])
        angels10 = np.outer(angels[:, 1], angels[:, 0])
        angels11 = np.outer(angels[:, 1], angels[:, 1])

        return np.stack([angels00, angels01, angels10, angels11], axis=2)

    def get_data(self):
        features = {}
        features['angels'] = self.protein.angels
        features['y_contact'] = self.target_pdb_cm  # shape l*l*1
        features['y_distance'] = self.target_pdb_dm  # shape l*l*1
        features['pssm'] = self.pssm  # shape l*l*44
        features['ec'] = self.plmc_score  # shape l*l*1
        features[
            'close_distance'] = self.closest_known_structure_dm  # shape l*l*2
        features[
            'mean_distance'] = self.mean_known_structures_dm  # shape l*l*2
        features['modeller_dm'] = self.modeller_dm  # shape l*l*1
        return features
