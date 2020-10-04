import matplotlib.pylab as plt
from keras import Sequential

from protein import Protein
from locator import figures_path
from utils import *
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, LSTM, Concatenate
from data_handler import MSA
import os
from tensorflow.keras import backend as K, Sequential
import tensorflow as tf
from tensorflow.keras import layers


def model_prediction_plot(msa, predictions, model_name):
    cover_prot = msa.metadata['structures'][msa.closest_known_strucutre]
    cover_protein = cover_prot[:4]
    chain = cover_prot[4]
    reference = Protein(cover_protein, chain).dm

    mask = msa.sequence_distance_matrix
    close_ind = mask < 6
    short_ind = (mask >= 6) & (mask < 12)
    medium_ind = (mask >= 12) & (mask < 24)
    long_ind = mask >= 24

    mask[close_ind] = 0
    mask[short_ind] = 1
    mask[medium_ind] = 2
    mask[long_ind] = 3

    target_dm = msa.target_pdb_cm
    aligned_reference = msa.closest_known_structure_dm[:, :, 0]
    target_dm = np.array(target_dm * mask, dtype=np.int)

    aligned_reference[~np.isnan(aligned_reference)] = aligned_reference[
        ~np.isnan(aligned_reference)] < 8.5
    aligned_reference[np.isnan((aligned_reference))] = 0
    aligned_reference_masked = np.array(aligned_reference * mask, dtype=np.int)

    reference[~np.isnan(reference)] = reference[~np.isnan(reference)] < 8.5
    reference[np.isnan(reference)] = 0
    predictions = np.array(predictions * mask, dtype=np.int)

    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    ax1.imshow(target_dm)
    ax1.title.set_text('Target')

    ax2.imshow(aligned_reference_masked)
    ax2.title.set_text('Aligned Reference')

    ax3.imshow(reference)
    ax3.title.set_text('Reference')

    ax4.imshow(predictions)
    ax4.title.set_text('%s' % model_name)

    model_fig_path = os.path.join(figures_path, model_name)
    if not os.path.exists(model_fig_path):
        os.mkdir(model_fig_path)

    plt.savefig(os.path.join(model_fig_path, '%s.png' % msa.target))


class ContactMapFinalPred(layers.Layer):
    def __init__(self):
        super(ContactMapFinalPred, self).__init__()

    def call(self, inputs):
        upper_triu = tf.matrix_band_part(inputs, 0, -1)
        diag = tf.matrix_band_part(inputs, 0, 0)
        print(upper_triu.shape)

        symetric_array = tf.add(upper_triu, tf.transpose(upper_triu))

        return symetric_array


class Angels_Outer(layers.Layer):
    def __init__(self, **kwargs):
        super(Angels_Outer, self).__init__(**kwargs)

    def call(self, inputs):
        phi = inputs[:, :, 0]
        psi = inputs[:, :, 1]

        phi_psi = tf.einsum("nu,nv->nuv", phi, psi)
        psi_psi = tf.einsum("nu,nv->nuv", psi, psi)
        phi_phi = tf.einsum("nu,nv->nuv", phi, phi)

        return tf.stack([phi_phi, psi_psi, phi_psi], axis=3)


def np_loss(y, y_hat):
    y_true = y.flatten()
    y_pred = y_hat.flatten()
    ce = -1 * np.mean(y_true * np.log(y_pred + 1e-07) +
                      (1 - y_true) * K.log(1 - y_pred + 1e-07))
    return ce


def contact_ce():
    def loss(y_true, y_pred):
        y = tf.cast(K.flatten(y_true), tf.float32)
        y_hat = tf.cast(K.flatten(y_pred), tf.float32)
        ce = -K.mean(y * K.log(y_hat + K.epsilon()) +
                     (1 - y) * K.log(1 - y_hat + K.epsilon()))
        return ce

    # Return a function
    return loss


# model = Sequential()
# # model.add(Conv2D(kernel_size=(1, 1), padding='same', filters=1))
# model.add(ContactMapFinalPred())
# model.compile(loss=contact_ce(),
#               optimizer='adam')
#
# x_train = np.random.random((1, 3, 3))
# y_train = x_train[:, :, 0]
# model.fit(x_train, y_train,
#           batch_size=1,
#           verbose=2)
# print(np.squeeze(x_train))
# print(model.predict(x_train))


class StructurePredictor():
    def __init__(self, name):
        self._name = name
        pass

    @property
    def model_path(self):

        model_path = '/cs/zbio/orzuk/projects/ContactMaps/data/MSA-Completion/models/%s' % self._name
        if not path.exists(model_path):
            print('model path is %s' % model_path)
            os.mkdir(model_path)
        return model_path

    def dataset(self, name='_proteins'):
        proteins_train = pd.read_csv(
            '/cs/zbio/orzuk/projects/ContactMaps/data/MSA-Completion/data/%s.csv'
            % name,
            index_col=0)
        all_targets = list(proteins_train.iloc[:, 0])
        known_structures_targets = []
        for target in all_targets:
            metadata_fname = '/cs/zbio/orzuk/projects/ContactMaps/data/MSA_Completion/msa_data/structures/%s/metadata' \
                             '.pkl' % target
            if os.path.isfile(metadata_fname):
                try:
                    meta_data = pkl_load(metadata_fname)
                except EOFError:
                    continue
                known_structures = meta_data[
                    'structures'] if 'structures' in meta_data else []

                if len(known_structures) > 1 and target in known_structures:
                    known_structures_targets.append(target)
        return known_structures_targets

    def get_predictions(self, msa, plot=False):
        """
        This function get the model ordered predictions
        :return: pd.DataFrame columns 'y' true values, 'y_hat' predictions,'sequence distance' aa sequence distance
        """
        raise NotImplementedError

    def evaluate_prediction(self, msa, plot=False):
        """

        :param target: (str) target protein to predict
        :return: (dict) ppv score for every distance category and every total predictions category
        """
        predictions = self.get_predictions(msa, plot=plot)
        predictions = predictions[predictions['sequence distance'] > 5]
        l = len(msa.protein.sequence)
        total_predictions_categories = [
            l, round(l / 2.0),
            round(l / 5.0), round(l / 10.0)
        ]
        ppv = {'S': [], 'M': [], 'L': []}
        n_pred = ['L', "L/2", 'L/5', 'L/10']
        predictions['distance_category'] = np.nan
        s_contacts = predictions['sequence distance'] <= 11
        m_contacts = np.logical_and(predictions['sequence distance'] > 11,
                                    predictions['sequence distance'] <= 24)
        l_contacts = predictions['sequence distance'] > 24
        indices = np.array(s_contacts, dtype=np.int) + np.array(
            m_contacts, dtype=np.int) + np.array(l_contacts, dtype=np.int)
        assert np.max(indices) == 1

        predictions['distance_category'][s_contacts] = 'S'
        predictions['distance_category'][m_contacts] = 'M'
        predictions['distance_category'][l_contacts] = 'L'

        for total_predictions in total_predictions_categories:
            for distance_category in ppv.keys():
                category_predictions = predictions[
                    predictions['distance_category'] == distance_category]
                n_predictions = category_predictions.shape[0]

                if n_predictions < total_predictions:
                    print('not enough predictions for categoty %s' %
                          distance_category)

                    diff = total_predictions - n_predictions
                    pseudo_predictions = pd.DataFrame(
                        {k: [0] * diff
                         for k in predictions.keys()})
                    category_predictions = pd.concat(
                        [category_predictions, pseudo_predictions],
                        axis=0,
                        ignore_index=True)
                top_predictions = category_predictions.head(
                    total_predictions)['y'].mean()
                if top_predictions == 0:
                    ppv[distance_category].append(1 / total_predictions)
                else:
                    ppv[distance_category].append(top_predictions)
        return pd.DataFrame(ppv, index=n_pred, columns=ppv.keys())

    def evaluate_dataset(self, targets, plot=False):
        ppv = {'S': [], 'M': [], 'L': []}
        n_pred = ['L', "L/2", 'L/5', 'L/10']
        dataset_predictions = []
        for target in targets:
            msa = MSA(target)
            logging.info("Evaluating %s" % target)
            if len(msa.known_structures) > 1:
                preds = self.evaluate_prediction(msa, plot=plot).values

                dataset_predictions.append(preds)

        return pd.DataFrame(np.mean(dataset_predictions, axis=0),
                            index=n_pred,
                            columns=ppv.keys())


class ReferencePredictor(StructurePredictor):
    def __init__(self, ref_func, name):
        super(ReferencePredictor, self).__init__(name=name)
        self._function_to_feature_map = {
            'close': 'closest_known_structure_dm_na',
            'mean': 'mean_known_structures_dm_na'
        }
        self._refrence_feature = self._function_to_feature_map[ref_func]

    def get_predictions(self, msa, plot=False):
        """

        :param msa (MSA): MSA object
        :return (pd.DataFrame) : columns 'y' true values, 'y_hat' predictions,'sequence distance' aa sequence distance
        """
        upper_triu = np.triu_indices(msa.target_seq_length_pdb)

        y_hat = getattr(msa, self._refrence_feature)[upper_triu]

        y = msa.target_pdb_cm[upper_triu]
        seq_dist = msa.sequence_distance_matrix[upper_triu]

        predictions = pd.DataFrame({
            'y': y.flatten(),
            'y_hat': y_hat.flatten(),
            'sequence distance': seq_dist.flatten()
        }).sort_values('y_hat', ascending=True)
        return predictions


class ModellerPredictor(StructurePredictor):
    def get_predictions(self, msa, plot=False):
        """

        :param msa (MSA): MSA object
        :return (pd.DataFrame) : columns 'y' true values, 'y_hat' predictions,'sequence distance' aa sequence distance
        """

        upper_triu = np.triu_indices(msa.target_seq_length_pdb)

        y_hat = msa.modeller_dm[upper_triu]
        y = msa.target_pdb_cm[upper_triu]
        seq_dist = msa.sequence_distance_matrix[upper_triu]
        predictions = pd.DataFrame({
            'y': y.flatten(),
            'y_hat': y_hat.flatten(),
            'sequence distance': seq_dist.flatten()
        }).sort_values('y_hat', ascending=True)
        return predictions

    def modeller_train(self):
        name = '150Pfam'
        proteins_train = pd.read_csv(
            '/cs/zbio/orzuk/projects/ContactMaps/data/MSA-Completion/data/%s.csv'
            % name,
            index_col=0)
        all_targets = list(proteins_train.iloc[:, 0])
        for target in all_targets:
            metadata_fanme = '/cs/zbio/orzuk/projects/ContactMaps/data/MSA_Completion/msa_data/structures/%s/metadata' \
                             '.pkl' % target
            if os.path.isfile(metadata_fanme):
                meta_data = pkl_load(metadata_fanme)
                known_structures = meta_data['structures']

                if len(known_structures) > 1 and target in known_structures:
                    msa = MSA(target=target)
                    print('target is %s' % target)
                    print(msa.modeller_dm)


class StatisticalPredictor(StructurePredictor):
    def __init__(self, n_epoch, batch_size, name, leaky, win_size=2):
        self._n_epoch = n_epoch
        self._batch_size = batch_size
        self._name = name
        self.leaky = leaky
        self._features_dim = {
            'full_pssm': 44,
            'plmc_score': 1,
            'closest_known_structure_dm': 2,
            'mean_known_structures_dm': 2,
            'stability_scores': 3,
            'outer_angels': 4
        }
        self._window_size = win_size
        self._checkpoint = path.join(self.model_path, 'checkpoint.hdf5')
        self._cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self._checkpoint, save_weights_only=True, verbose=1)
        self._weights = path.join(self.model_path, 'weights.h5')
        self._model_data = path.join(self.model_path, 'net.h5')

        self.net = self.get_net()

    @property
    def _features(self):
        raise NotImplementedError

    def get_random_train_data(self):
        raise NotImplementedError

    def get_model_data(self, msa):
        x = []
        for f in self._features:
            # start = time.time()
            feature = getattr(msa, f)
            # end = time.time()
            # print('Generating %s took %s' % (f, str(end - start)))
            if len(feature.shape) == 3:
                x.append(feature)
            else:
                x.append(np.expand_dims(feature, axis=2))
        pairwise = np.expand_dims(np.concatenate(x, axis=2), axis=0)
        contact = msa.target_pdb_dm if self.net.loss == 'mean_squared_error' else msa.target_pdb_cm

        x, y = pairwise, contact.reshape(1, contact.shape[0], contact.shape[1],
                                         1)

        return x, y

    def get_full_model_data(self, msa):
        x = []
        for f in self._features:
            start = time.time()
            feature = getattr(msa, f)
            end = time.time()
            # print('Generating %s took %s' % (f, str(end - start)))
            if len(feature.shape) == 3:
                x.append(feature)
            else:
                x.append(np.expand_dims(feature, axis=2))
        pairwise = np.expand_dims(np.concatenate(x, axis=2), axis=0)
        contact_map = msa.target_pdb_dm if self.net.loss == 'mean_squared_error' else msa.target_pdb_cm
        contact_map = contact_map.reshape(1, contact_map.shape[0],
                                          contact_map.shape[1], 1)
        angels_ref = np.expand_dims(msa.closest_known_structure_angels, axis=0)
        angels_true = np.expand_dims(msa.target_angels, axis=0)

        x, y = {
            'angels_ref': angels_ref,
            'pairwise': pairwise
        }, {
            'angels_pred': angels_true,
            'conv': contact_map
        }

        return x, y

    def data_generator(self, training_set, leaky=False):
        # start = (batch_number - 1) * self._batch_size
        # end = batch_number * self._batch_size
        # for protein in _proteins[start:end]:
        #     msa = MSA(protein)
        #     if len(msa.known_structures) > 1:
        #         yield self.get_model_data(msa)
        yielded = 0
        while yielded < self._batch_size:
            try:
                protein = str(np.random.choice(training_set, 1)[0], 'utf-8')
            except TypeError:
                protein = np.random.choice(training_set, 1)[0]

            msa = MSA(protein)
            if len(msa.known_structures
                   ) > 1 and protein in msa.known_structures:
                if leaky:

                    yield self.get_model_data(msa)
                    yielded += 1

                else:
                    try:
                        yield self.get_full_model_data(msa)
                        yielded += 1
                    except ValueError:
                        continue

    @timefunc
    def model_dataset(self, training_set, leaky=False):
        # dataset = tf.data.Dataset.from_generator(self.data_generator,
        #                                          args=(_proteins, batch_number),
        #                                          output_types=(tf.int64, tf.int64),
        #                                          output_shapes=(tf.TensorShape([None] * len(self.net.input_shape)),
        #                                                         tf.TensorShape([None] * len(self.net.input_shape))))
        output_types = {
            'angels_ref': tf.int64,
            'pairwise': tf.int64
        }, {
            'angels_pred': tf.int64,
            'conv': tf.int64
        }
        output_shapes = {
            'angels_ref': [None, None, 4],
            'pairwise': [None] * 4
        }, {
            'angels_pred': [None, None, 2],
            'conv': [None] * 4
        }
        if leaky:
            output_types = (tf.float64, tf.float64)
            output_shapes = (tf.TensorShape([None] *
                                            len(self.net.input_shape)),
                             tf.TensorShape([None] *
                                            len(self.net.input_shape)))
        dataset = tf.data.Dataset.from_generator(self.data_generator,
                                                 args=[training_set, leaky],
                                                 output_types=output_types,
                                                 output_shapes=output_shapes)

        return dataset.repeat()

    @timefunc
    def train(self, leaky=False):

        for e in range(self._n_epoch):
            #     for batch_number in range(1, n_batches + 1):
            ds = self.model_dataset(self.dataset(), leaky=leaky)

            self.net.fit(ds,
                         verbose=2,
                         epochs=1,
                         shuffle=False,
                         steps_per_epoch=self._batch_size,
                         callbacks=[self._cp_callback])
            # self.net.save_weights(self._weights)

    def get_predictions(self, msa, plot=False):
        """

        :param msa (MSA): MSA object
        :return (pd.DataFrame) : columns 'y' true values, 'y_hat' predictions,'sequence distance' aa sequence distance
        """

        upper_triu = np.triu_indices(msa.target_seq_length_pdb)

        input, labels = self.get_model_data(
            msa) if self.leaky else self.get_full_model_data(msa)
        contact = labels if self.leaky else labels['conv']

        x = input if self.leaky else input['pairwise']

        y = np.squeeze(np.array(contact))[upper_triu]

        predictions = self.net.predict(input)

        contact_pred_matrix = np.squeeze(
            predictions) if self.leaky else np.squeeze(predictions[1])

        contact_pred = contact_pred_matrix[upper_triu]

        if np.max(y) > 1:
            y = np.array(y < 8.5, dtype=np.int)
            contact_pred = -1 * np.exp(contact_pred)
            contact_pred_matrix = np.array(contact_pred_matrix < 8.5,
                                           dtype=np.int)

        else:
            bad_ind = msa.sequence_distance_matrix <= 5
            short_ind = np.logical_and(msa.sequence_distance_matrix > 5,
                                       msa.sequence_distance_matrix <= 11)
            medium_ind = np.logical_and(msa.sequence_distance_matrix > 11,
                                        msa.sequence_distance_matrix < 24)
            long_ind = msa.sequence_distance_matrix >= 24

            category_contacts = msa.target_seq_length_pdb * 3

            short_predictions = contact_pred_matrix[short_ind]
            short_threshold = np.partition(
                short_predictions.flatten(),
                -category_contacts)[-category_contacts]

            medium_predictions = contact_pred_matrix[medium_ind]
            medium_threshold = np.partition(
                medium_predictions.flatten(),
                -category_contacts)[-category_contacts]

            long_predictions = contact_pred_matrix[long_ind]
            long_threshold = np.partition(
                long_predictions.flatten(),
                -category_contacts)[-category_contacts]

            contact_pred_matrix[bad_ind] = 0
            contact_pred_matrix[long_ind] = np.array(
                contact_pred_matrix[long_ind] >= long_threshold, dtype=np.int)
            contact_pred_matrix[medium_ind] = np.array(
                contact_pred_matrix[medium_ind] >= medium_threshold,
                dtype=np.int)
            contact_pred_matrix[short_ind] = np.array(
                contact_pred_matrix[short_ind] >= short_threshold,
                dtype=np.int)

        if plot:
            model_prediction_plot(msa=msa,
                                  predictions=contact_pred_matrix,
                                  model_name=self._name)

        plmc = x[:, :, :, 4] if self.leaky else x[:, :, :, 0]
        plmc = np.squeeze(plmc)[upper_triu]
        seq_dist = msa.sequence_distance_matrix[upper_triu]
        predictions = pd.DataFrame({
            'y': y.flatten(),
            'y_hat': contact_pred.flatten(),
            'plmc': plmc.flatten(),
            'sequence distance': seq_dist
        })
        return predictions.sort_values('y_hat', ascending=False)

    def get_net(self):
        net = self.create_net()

        if os.path.isfile(self._weights):
            # net = tf.keras.models.load_model(self._model_data, custom_objects={'loss':contact_ce()})
            net.load_weights(self._checkpoint)

        return net

    def create_net(self):
        raise NotImplementedError


class RefAngelsPlmc(StatisticalPredictor):
    @property
    def _features(self):
        return ['plmc_score', 'closest_known_structure_dm']

    def create_net(self):
        PAIRWISE_FEATURES = np.sum(
            [self._features_dim[f] for f in self._features])

        angels_ref = Input(shape=(None, 4), name='angels_ref')
        angels_pred = LSTM(2, return_sequences=True,
                           name='angels_pred')(angels_ref)

        angels_outer = Angels_Outer()(angels_pred)
        pairwise = Input(shape=(None, None, PAIRWISE_FEATURES),
                         name='pairwise')
        full_pairwise = Concatenate(axis=3)([pairwise, angels_outer])
        conv1 = Conv2D(kernel_size=(20, 20),
                       padding='same',
                       filters=10,
                       activation='tanh',
                       name='conv1')(full_pairwise)
        conv2 = Conv2D(kernel_size=(20, 20),
                       padding='same',
                       filters=10,
                       activation='tanh',
                       name='conv2')(conv1)
        conv3 = Conv2D(kernel_size=(20, 20),
                       padding='same',
                       filters=10,
                       activation='tanh',
                       name='conv3')(conv2)
        conv = Conv2D(kernel_size=(20, 20),
                      padding='same',
                      filters=1,
                      activation='sigmoid',
                      name='conv')(conv3)

        net = Model(inputs=[angels_ref, pairwise], outputs=[angels_pred, conv])
        net.compile(loss={
            'angels_pred': 'mean_squared_error',
            'conv': contact_ce()
        },
                    loss_weights={
                        'angels_pred': 1.,
                        'conv': 0.2
                    },
                    optimizer='adam')
        return net

    def get_loss_value(self, x, y):
        predictions = self.net.predict(x)
        angels, contact = y['angels_pred'], y['conv']
        contact_hat = predictions[1].flatten()
        angels_hat = predictions[0]
        contact = contact.flatten()
        ce = -np.mean(contact * np.log(contact_hat + K.epsilon()) +
                      (1 - contact) * np.log(1 - contact_hat + K.epsilon()))
        mse = np.mean((angels - angels_hat)**2)

        return pd.DataFrame({'ce': ce, 'msa': mse}, index=[0])


class RefAngelsPlmcContinous(StatisticalPredictor):
    @property
    def _features(self):
        return ['plmc_score', 'closest_known_structure_dm']

    def create_net(self):
        PAIRWISE_FEATURES = np.sum(
            [self._features_dim[f] for f in self._features])

        angels_ref = Input(shape=(None, 4), name='angels_ref')
        angels_pred = LSTM(2, return_sequences=True,
                           name='angels_pred')(angels_ref)
        angels_outer = Angels_Outer()(angels_pred)
        pairwise = Input(shape=(None, None, PAIRWISE_FEATURES),
                         name='pairwise')
        full_pairwise = Concatenate(axis=3)([pairwise, angels_outer])
        conv1 = Conv2D(kernel_size=(20, 20),
                       padding='same',
                       filters=10,
                       activation='tanh',
                       name='conv1')(full_pairwise)
        conv2 = Conv2D(kernel_size=(20, 20),
                       padding='same',
                       filters=10,
                       activation='tanh',
                       name='conv2')(conv1)
        conv3 = Conv2D(kernel_size=(20, 20),
                       padding='same',
                       filters=10,
                       activation='tanh',
                       name='conv3')(conv2)
        conv = Conv2D(kernel_size=(20, 20),
                      padding='same',
                      filters=1,
                      activation='relu',
                      name='conv')(conv3)

        net = Model(inputs=[angels_ref, pairwise], outputs=[angels_pred, conv])
        net.compile(loss={
            'angels_pred': 'mean_squared_error',
            'conv': 'mean_squared_error'
        },
                    loss_weights={
                        'angels_pred': 1.,
                        'conv': 0.2
                    },
                    optimizer='adam')
        return net

    def get_loss_value(self, x, y):
        predictions = self.net.predict(x)
        angels, contact = y['angels_pred'], y['conv']
        contact_hat = predictions[1].flatten()
        angels_hat = predictions[0]
        contact = contact.flatten()
        ce = np.mean((contact_hat - contact)**2)
        mse = np.mean((angels - angels_hat)**2)

        return pd.DataFrame({'ce': ce, 'msa': mse}, index=[0])


class RefLeakyAngelsPlmc(StatisticalPredictor):
    @property
    def _features(self):
        return ['outer_angels', 'plmc_score', 'closest_known_structure_dm']

    def create_net(self):
        raise NotImplementedError

    def get_random_train_data(self):
        l = 5
        s = self.net.input_shape[3]
        shp = (1, l, l, s)
        x = np.random.random(shp)
        y = np.random.choice([0, 1], size=(1, l, l, 1), p=[1. / 3, 2. / 3])
        return x, y

    def get_loss_value(self, x, y):
        y_hat = self.net.predict(x).flatten()
        y = y.flatten()
        ce = -np.mean(y * np.log(y_hat + K.epsilon()) +
                      (1 - y) * np.log(1 - y_hat + K.epsilon()))

        return ce

    def create_net(self):
        input_dim = np.sum([self._features_dim[f] for f in self._features])
        input = Input(shape=(None, None, input_dim), name='pairwise')
        conv1 = Conv2D(kernel_size=(20, 20),
                       padding='same',
                       filters=10,
                       activation='tanh',
                       name='conv1')(input)
        conv2 = Conv2D(kernel_size=(20, 20),
                       padding='same',
                       filters=10,
                       activation='tanh',
                       name='conv2')(conv1)
        conv3 = Conv2D(kernel_size=(20, 20),
                       padding='same',
                       filters=10,
                       activation='tanh',
                       name='conv3')(conv2)
        conv = Conv2D(kernel_size=(20, 20),
                      padding='same',
                      filters=1,
                      activation='sigmoid',
                      name='conv')(conv3)
        net = Model(input, conv)

        net.compile(loss=contact_ce(), optimizer='adam')
        return net


class RefLeaky(StatisticalPredictor):
    @property
    def _features(self):
        return ['outer_angels', 'plmc_score', 'closest_known_structure_dm']

    def create_net(self):
        raise NotImplementedError

    def get_random_train_data(self):
        l = 5
        s = self.net.input_shape[3]
        shp = (1, l, l, s)
        x = np.random.random(shp)
        y = np.random.choice([0, 1], size=(1, l, l, 1), p=[1. / 3, 2. / 3])
        return x, y

    def get_loss_value(self, x, y):
        y_hat = self.net.predict(x).flatten()
        y = y.flatten()
        ce = -np.mean(y * np.log(y_hat + K.epsilon()) +
                      (1 - y) * np.log(1 - y_hat + K.epsilon()))

        return ce

    def create_net(self):
        input_dim = np.sum([self._features_dim[f] for f in self._features])
        input = Input(shape=(None, None, input_dim), name='pairwise')
        conv = Conv2D(kernel_size=(5, 5),
                      padding='same',
                      filters=1,
                      activation='sigmoid',
                      name='conv')(input)
        net = Model(input, conv)

        net.compile(loss=contact_ce(), optimizer='adam')
        return net


class RefLeakyAngelsPlmcContinous(StatisticalPredictor):
    @property
    def _features(self):
        return ['outer_angels', 'plmc_score', 'closest_known_structure_dm']

    def get_random_train_data(self):
        l = 5
        s = self.net.input_shape[3]
        shp = (1, l, l, s)
        x = np.random.random(shp)
        y = np.random.choice([0, 1], size=(1, l, l, 1), p=[1. / 3, 2. / 3])
        return x, y

    def get_loss_value(self, x, y):
        y_hat = self.net.predict(x).flatten()
        y = y.flatten()
        ce = np.mean((y - y_hat)**2)

        return ce

    def create_net(self):
        input_dim = np.sum([self._features_dim[f] for f in self._features])
        input = Input(shape=(None, None, input_dim), name='pairwise')
        conv1 = Conv2D(kernel_size=(20, 20),
                       padding='same',
                       filters=10,
                       activation='tanh',
                       name='conv1')(input)
        conv2 = Conv2D(kernel_size=(20, 20),
                       padding='same',
                       filters=10,
                       activation='tanh',
                       name='conv2')(conv1)
        conv3 = Conv2D(kernel_size=(20, 20),
                       padding='same',
                       filters=10,
                       activation='tanh',
                       name='conv3')(conv2)
        conv = Conv2D(kernel_size=(20, 20),
                      padding='same',
                      filters=1,
                      activation='relu',
                      name='conv')(conv3)
        net = Model(input, conv)

        net.compile(loss='mean_squared_error', optimizer='adam')
        return net


# class RefLeakyAngelsPlmcConvBin(RefLeakyAngelsPlmc):
#
#     def create_net(self):
#         input_dim = np.sum([self._features_dim[f] for f in self._features])
#         input = Input(shape=(None, None, input_dim), name='pairwise')
#         conv = Conv2D(kernel_size=(20, 20), padding='same', filters=10, activation='tanh', name='contact')(input)
#         net = Model(input, conv)
#
#         net.compile(loss=contact_ce(), optimizer='adam')
#         return net
#
#
# class RefLeakyAngelsPlmcLogisticBin(RefLeakyAngelsPlmc):
#
#     def create_net(self):
#         input_dim = np.sum([self._features_dim[f] for f in self._features])
#         input = Input(shape=(None, None, input_dim), name='pairwise')
#         conv = Conv2D(kernel_size=(1, 1), padding='same', filters=1, activation='sigmoid', name='contact')(input)
#         net = Model(inputs=[input], outputs=[conv])
#         net.compile(loss='binary_crossentropy', optimizer='adam')
#         return net
#
#
# class RefLeakyAngelsPlmcConvBinDeep(RefLeakyAngelsPlmc):
#
#     def create_net(self):
#         input_dim = np.sum([self._features_dim[f] for f in self._features])
#         input = Input(shape=(None, None, input_dim), name='pairwise')
#         conv1 = Conv2D(kernel_size=(3, 3), padding='same', filters=10)(input)
#         conv2 = Conv2D(kernel_size=(5, 5), padding='same', filters=1, activation='sigmoid', name='contact')(conv1)
#         net = Model(inputs=[input], outputs=[conv2])
#         net.compile(loss=contact_ce(), optimizer='adam')
#         return net
#
#
# class RefLeakyAngelsPlmcConvCont(RefLeakyAngelsPlmc):
#
#     def create_net(self):
#         input_dim = np.sum([self._features_dim[f] for f in self._features])
#         input = Input(shape=(None, None, input_dim), name='pairwise')
#         conv = Conv2D(kernel_size=(3, 3), padding='same', filters=1, name='contact', activation='relu')(input)
#         net = Model(inputs=[input], outputs=[conv])
#         net.compile(loss='mean_squared_error', optimizer='adam')
#         return net
#
#
# class RefLeakyAngelsPlmcConvContDeep(RefLeakyAngelsPlmc):
#
#     def create_net(self):
#         input_dim = np.sum([self._features_dim[f] for f in self._features])
#         input = Input(shape=(None, None, input_dim), name='pairwise')
#         conv1 = Conv2D(kernel_size=(3, 3), padding='same', filters=10)(input)
#         conv2 = Conv2D(kernel_size=(5, 5), padding='same', filters=1, name='contact', activation='relu')(conv1)
#         net = Model(inputs=[input], outputs=[conv2])
#         net.compile(loss='mean_squared_error', optimizer='adam')
#         return net
#
#
# class LSTMLeakyAngelsPlmcRef(StatisticalPredictor):
#
#     @property
#     def _features(self):
#         return ['plmc_score', 'closest_known_structure_dm', 'outer_angels']
#
#     def get_model_data(self, msa):
#         x, y = super().get_model_data(msa)
#         x = np.squeeze(x, axis=0)
#         y = np.squeeze(y)
#         pairs_x = list(itertools.combinations(range(x.shape[1]), 2))
#         pairs = itertools.combinations(range(x.shape[1]), 2)
#         pairs_y = tuple(zip(*pairs))
#         x = np.expand_dims(get_conv_data_x(x=x, pairs=pairs_x, window_size=self._window_size), axis=0)
#         y = np.expand_dims(np.expand_dims(y[pairs_y], axis=0), axis=2)
#         # print('Generated data')
#         return x, y
#
#     def get_predictions(self, msa):
#         """
#
#         :param msa (MSA): MSA object
#         :return (pd.DataFrame) : columns 'y' true values, 'y_hat' predictions,'sequence distance' aa sequence distance
#         """
#         x, y = self.get_model_data(msa)
#         y = np.array(y)
#
#         y_hat = self.net.predict(x)
#         pairs_x = list(itertools.combinations(range(msa.sequence_distance_matrix.shape[1]), 2))
#         pairs_x = list(zip(*pairs_x))
#
#         predictions = pd.DataFrame({'y': y.flatten(),
#                                     'y_hat': y_hat.flatten(),
#                                     'sequence distance': msa.sequence_distance_matrix[pairs_x]}).sort_values('y_hat',
#                                                                                                              ascending=False)
#         return predictions
#
#     # def train(self, msa, history=None):
#     #     x, y, y_dist = self.get_model_data(msa)
#     #     y = np.array(y).flatten()
#     #     y_dist = np.array(y_dist).flatten()
#     #
#     #     if self.net.loss == 'binary_crossentropy':
#     #         y = y.reshape(1, y.shape[0], 1)
#     #     if self.net.loss == 'mean_squared_error':
#     #         y = y_dist.reshape(1, y.shape[0], 1)
#     #
#     #     if history is not None:
#     #         self.net.fit(x, y, verbose=2, epochs=self._n_epoch, shuffle=False, batch_size=self._batch_size,
#     #                      callbacks=[history])
#     #     else:
#     #         self.net.fit(x, y, verbose=2, epochs=self._n_epoch, shuffle=False, batch_size=self._batch_size)
#     #
#     #     self.net.save(path.join(self.model_path, 'net.h5'))
#     #     self.net.save_weights(path.join(self.model_path, 'weights.h5'))
#
#
# class LSTMLeakyAngelsPlmcRefBin(LSTMLeakyAngelsPlmcRef):
#
#     def create_net(self):
#         input_dim = np.sum([self._features_dim[f] for f in self._features]) * (2 * self._window_size + 1) ** 2
#
#         evolutionairy_sequence = Input(shape=(None, input_dim), name='evolutionairy_sequence')
#         lstm = LSTM(1, input_shape=(None, input_dim), name='b-lstm',
#                     return_sequences=True)(evolutionairy_sequence)
#         # conv = Conv1D(kernel_size=2, input_shape=(None, None, 2),
#         #               padding='same', filters=1, activation='sigmoid')(lstm)
#
#         net = Model(inputs=[evolutionairy_sequence], outputs=[lstm])
#         net.compile(loss='binary_crossentropy', optimizer='adam')
#         return net
#
#
# class LSTMLeakyAngelsPlmcRefBinDeep(LSTMLeakyAngelsPlmcRef):
#
#     def create_net(self):
#         input_dim = np.sum([self._features_dim[f] for f in self._features]) * (2 * self._window_size + 1) ** 2
#
#         evolutionairy_sequence = Input(shape=(None, input_dim), name='evolutionairy_sequence')
#         lstm = Bidirectional(LSTM(5, input_shape=(None, input_dim), name='b-lstm',
#
#                                   return_sequences=True))(evolutionairy_sequence)
#         conv1 = Conv1D(kernel_size=10, input_shape=(None, None, 2),
#                        padding='same', filters=5)(lstm)
#         conv2 = Conv1D(kernel_size=2, input_shape=(None, None, 2),
#                        padding='same', filters=1, activation='sigmoid')(conv1)
#
#         net = Model(inputs=[evolutionairy_sequence], outputs=[conv2])
#         net.compile(loss='binary_crossentropy', optimizer='adam')
#         return net
#
#
# class LSTMLeakyAngelsPlmcRefCont(LSTMLeakyAngelsPlmcRef):
#
#     def create_net(self):
#         input_dim = np.sum([self._features_dim[f] for f in self._features]) * (2 * self._window_size + 1) ** 2
#
#         evolutionairy_sequence = Input(shape=(None, input_dim), name='evolutionairy_sequence')
#         lstm = Bidirectional(LSTM(1, input_shape=(None, input_dim), name='b-lstm',
#
#                                   return_sequences=True))(evolutionairy_sequence)
#         conv = Conv1D(kernel_size=2, input_shape=(None, None, 2),
#                       padding='same', filters=1, activation='relu')(lstm)
#
#         net = Model(inputs=[evolutionairy_sequence], outputs=[conv])
#         net.compile(loss='mean_squared_error', optimizer='adam')
#         return net
#
#
# class LSTMLeakyAngelsPlmcRefContDeep(LSTMLeakyAngelsPlmcRef):
#
#     def create_net(self):
#         input_dim = np.sum([self._features_dim[f] for f in self._features]) * (2 * self._window_size + 1) ** 2
#
#         evolutionairy_sequence = Input(shape=(None, input_dim), name='evolutionairy_sequence')
#         lstm = Bidirectional(LSTM(5, input_shape=(None, input_dim), name='b-lstm',
#
#                                   return_sequences=True))(evolutionairy_sequence)
#         conv1 = Conv1D(kernel_size=10, input_shape=(None, None, 2),
#                        padding='same', filters=5)(lstm)
#         conv2 = Conv1D(kernel_size=2, input_shape=(None, None, 2),
#                        padding='same', filters=1, activation='relu')(conv1)
#
#         net = Model(inputs=[evolutionairy_sequence], outputs=[conv2])
#         net.compile(loss='mean_squared_error', optimizer='adam')
#         return net
