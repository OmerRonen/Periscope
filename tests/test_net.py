import unittest
import tempfile

from periscope.net.contact_map import ContactMapEstimator
from periscope.net.params import NetParams
from periscope.utils.constants import LOCAL, ARCHS, FEATURES, PATHS


class TestProteinNet(unittest.TestCase):

    def test_train_cnn_random(self):
        tempdir = tempfile.TemporaryDirectory(dir=PATHS.models)
        model_name = tempdir.name.split('/')[-1]
        params = NetParams.generate_net_params(name=model_name,
                                               conv_features=[FEATURES.ccmpred, FEATURES.evfold],
                                               arch=ARCHS.periscope2,
                                               k=10,
                                               save_summary_steps=2,
                                               save_checkpoints_steps=20,
                                               batch_size=1,
                                               epochs=3,
                                               num_bins=2,
                                               num_channels=10,
                                               deep_projection=False,
                                               filter_shape=(30, 30),
                                               dilation=1,
                                               num_layers=4,
                                               train_dataset=None,
                                               eval_dataset=None)

        if LOCAL:
            cnn_pred = ContactMapEstimator(params)
            cnn_pred.train_and_evaluate()
            # shutil.rmtree(tempdir.name, ignore_errors=True)

    def test_train_cnn_live_0(self):
        tempdir = tempfile.TemporaryDirectory(dir=PATHS.models)
        model_name = tempdir.name.split('/')[-1]
        params = NetParams.generate_net_params(
            name=model_name,
            conv_features=[FEATURES.reference_dm, FEATURES.ccmpred, FEATURES.evfold],
            arch=ARCHS.conv,
            save_summary_steps=1,
            save_checkpoints_steps=1,
            epochs=1,
            num_bins=2,
            batch_size=1,
            num_channels=10,
            filter_shape=(5, 5),
            dilation=1,
            num_layers=2,
            k=9,
            lr=0.001,
            train_dataset='testing',
            eval_dataset='testing',
            test_dataset='testing')

        if not LOCAL:
            cnn_pred = ContactMapEstimator(params)
            cnn_pred.train_and_evaluate()
            # shutil.rmtree(tempdir.name, ignore_errors=True)

    def test_train_cnn_live_1(self):
        tempdir = tempfile.TemporaryDirectory(dir=PATHS.models)
        model_name = tempdir.name.split('/')[-1]

        params = NetParams.generate_net_params(
            name=model_name,
            conv_features=[FEATURES.ccmpred, FEATURES.reference_dm, FEATURES.evfold],
            arch=ARCHS.periscope2,
            save_summary_steps=1,
            save_checkpoints_steps=1,
            epochs=1,
            num_bins=2,
            batch_size=1,
            num_channels=10,
            deep_projection=True,
            filter_shape=(5, 5),
            dilation=1,
            num_layers=2,
            k=10,
            lr=0.001,
            train_dataset='testing',
            eval_dataset='testing',
            test_dataset='testing',
            templates_dropout=0.9)

        if not LOCAL:
            cnn_pred = ContactMapEstimator(params)
            cnn_pred.train_and_evaluate()
            # shutil.rmtree(tempdir.name, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
