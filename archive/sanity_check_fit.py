import logging
import os
import shutil

import numpy as np
import tensorflow as tf

from .cm_predictor import ContactMapPredictor
from .protein_net import NetParams

logging.getLogger('parso.cache').disabled = True
logging.getLogger('parso.cache.pickle').disabled = True
logging.getLogger('parso.python.diff').disabled = True

logger_tf = tf.get_logger()
logger_tf.propagate = False
if os.path.exists('models/test_huji_ce_dm'):
    shutil.rmtree('models/test_huji_ce_dm')
if os.path.exists('models/test_huji_mse_dm'):
    shutil.rmtree('models/test_huji_mse_dm')

params = NetParams.generate_net_params(path='models/test_huji_ce_dm',
                                       features=['plmc_score', 'reference_dm'],
                                       save_summary_steps=500,
                                       save_checkpoints_steps=500,
                                       epochs=5000,
                                       steps=1,
                                       num_bins=2,
                                       num_channels=5,
                                       filter_shape=(7, 7),
                                       num_layers=3,
                                       lr=0.0001,
                                       train_dataset='testing',
                                       eval_dataset='testing')

params_2 = NetParams.generate_net_params(
    path='models/test_huji_mse_dm',
    features=['plmc_score', 'reference_dm'],
    save_summary_steps=500,
    save_checkpoints_steps=500,
    epochs=5000,
    steps=1,
    num_bins=1,
    num_channels=5,
    filter_shape=(7, 7),
    num_layers=3,
    lr=0.0001,
    train_dataset='testing',
    eval_dataset='testing')

if __name__ == '__main__':
    cnn_pred = ContactMapPredictor(params)
    cnn_pred.train_and_evaluate()
    pred_gen = cnn_pred.get_predictions_generator()
    x = next(pred_gen)
    np.savetxt('models/test_huji_ce/predictions.csv',
               np.squeeze(x[..., 0]),
               delimiter=',')
    cm = next(cnn_pred.predict_data_manager.data_generator())['contact_map']
    np.savetxt('models/test_huji_ce/ground_truth.csv',
               np.squeeze(cm[..., 0]),
               delimiter=',')

    cnn_pred = ContactMapPredictor(params_2)
    cnn_pred.train_and_evaluate()
    pred_gen = cnn_pred.get_predictions_generator()
    x = next(pred_gen)
    np.savetxt('models/test_huji_mse/predictions.csv',
               np.squeeze(x[..., 0]),
               delimiter=',')
    cm = next(cnn_pred.predict_data_manager.data_generator())['contact_map']
    np.savetxt('models/test_huji_mse/ground_truth.csv',
               np.squeeze(cm[..., 0]),
               delimiter=',')
