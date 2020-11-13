import os
import logging

from ..utils.constants import PATHS
from ..utils.utils import yaml_load, yaml_save

logging.basicConfig(level=logging.INFO)

LOGGER = logging.getLogger(__name__)


class NetParams:
    def __init__(self, params_file, **kwargs):
        self.params_file = params_file
        self.params = self.load() if os.path.isfile(
            params_file) else self.generate_net_params(**kwargs)

    def load(self):
        params = yaml_load(self.params_file)
        params['net']['data']['k'] = params['net']['data'].get('k', None)

        params['train']['test_dataset'] = params['train'].get(
            'test_dataset', params['train']['eval_dataset'])

        params['net']['architecture']['arch'] = params['net'][
            'architecture'].get('arch', 'conv')

        params['net']['architecture']['ms'] = params['net']['architecture'].get('ms', {"deep_projection": False})

        params['train']['batch_size'] = params['train'].get('batch_size', 1)
        params['train']['templates_dropout'] = params['train'].get('templates_dropout', 0)

        if 'features' in params['net']['data'] and 'conv_features' not in params['net']['data']:
            params['net']['data']['conv_features'] = params['net']['data'][
                'features']
            del params['net']['data']['features']

        return params

    def save(self):
        LOGGER.info('saving to %s' % self.params_file)
        yaml_save(self.params_file, self.params)

    @staticmethod
    def generate_net_params(conv_features,
                            name,
                            arch,
                            batch_size=1,
                            epochs=20,
                            num_bins=2,
                            num_layers=12,
                            num_channels=8,
                            filter_shape=6 * [(5, 5)] + 6 * [(10, 10)],
                            dilation=1,
                            deep_projection=False,
                            k=None,
                            lr=0.0001,
                            save_summary_steps=500,
                            save_checkpoints_steps=2500,
                            train_dataset='train',
                            eval_dataset='eval',
                            test_dataset='pfam',
                            templates_dropout=0):
        """Generates the parameters for convolution contact map predictor

        Args:
            conv_features (list[str]): conv_features names
            name (str): model name
            arch (str): name of the net arch
            batch_size (int): batch size for training
            epochs (int): Number of training epochs
            num_bins (int): number of bins to predict 1 is regression more than 1 is clasification
            num_layers (int): number of hidden layers
            num_channels (int): number of channels for every conv operation
            filter_shape (Union[tuple[int, int], list[tuple]]): shape of the convolution filter
            dilation (Union[int, list[int]): dilation factor for conv layer
            deep_projection (bool): if true we use a deep operation for projection, othersie linear.
            k (int): number of msa structures to sample
            lr (float): learning rate for the optimizer
            save_summary_steps (int): Number of steps between two summary saves
            save_checkpoints_steps (int): number of steps to save model checkpoint
            train_dataset (Union[str, None]): name of the training dataset
            eval_dataset (Union[str, None]): name of the eval dataset
            test_dataset (Union[str, None]): name of the test dataset
            templates_dropout (float): probability to use all zero array instead of templates.

        Returns:
            dict: the parameters dict

        """

        net = {
            'architecture': {
                'conv': {
                    'num_bins': num_bins,
                    'num_layers': num_layers,
                    'num_channels': num_channels,
                    'filter_shape': filter_shape,
                    'dilation': dilation,
                },
                'ms':
                    {"deep_projection": deep_projection},

                'arch': arch,
            },
            'data': {
                'conv_features': conv_features,
                'k': k,
            }
        }
        train = {
            'epochs': epochs,
            'batch_size': batch_size,
            "templates_dropout":templates_dropout,
            'path': os.path.join(PATHS.models, name),
            'in_train': {
                'save_summary_steps': save_summary_steps,
                'save_checkpoints_steps': save_checkpoints_steps
            },
            'opt': {
                'lr': lr
            },
            'train_dataset': train_dataset,
            'eval_dataset': eval_dataset,
            'test_dataset': test_dataset
        }

        return {'net': net, 'train': train}
