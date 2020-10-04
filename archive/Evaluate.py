import logging
import os

from archive.NET import (ReferencePredictor, ModellerPredictor, RefAngelsPlmc,
                         RefAngelsPlmcContinous, RefLeakyAngelsPlmc,
                         RefLeakyAngelsPlmcContinous)

# logging.getLogger().setLevel(logging.INFO)

EPOCHS = 2000
BATCH_SIZE = 10

today = '2020-02-09'
tday2 = '2020-02-08'
#
models = {
    'RefAngelsPlmc':
    RefAngelsPlmc(batch_size=BATCH_SIZE,
                  n_epoch=EPOCHS,
                  name='RefAngelsPlmc_%s' % today,
                  leaky=False),
    'RefAngelsPlmcContinous':
    RefAngelsPlmcContinous(batch_size=BATCH_SIZE,
                           n_epoch=EPOCHS,
                           name='RefAngelsPlmcContinous_%s' % today,
                           leaky=False),
    'RefLeakyAngelsPlmc':
    RefLeakyAngelsPlmc(batch_size=BATCH_SIZE,
                       n_epoch=EPOCHS,
                       name='RefLeakyAngelsPlmc_%s' % tday2,
                       leaky=True),
    'RefLeakyAngelsPlmcContinous':
    RefLeakyAngelsPlmcContinous(batch_size=BATCH_SIZE,
                                n_epoch=EPOCHS,
                                name='RefLeakyAngelsPlmcContinous_%s' % today,
                                leaky=True),
    'ReferencePredictor':
    ReferencePredictor(ref_func='close', name='ReferencePredictor_%s' % today),
    'ModellerPredictor':
    ModellerPredictor(name='ModellerPredictor_%s' % today)
}


def evaluate(models, date):
    for name, model in models.items():
        evaluation = model.evaluate_dataset(targets=model.dataset('150Pfam'),
                                            plot=True)
        logging.info("%s evaluation is:\n" % name)
        logging.info(evaluation)
        evaluation.to_csv(
            os.path.join(model.model_path, '150Pfam_%s.csv' % date))


if __name__ == '__main__':
    today = '2020-02-09'
    # ModellerPredictor(name='ModellerPredictor_%s' % today).modeller_train()
    # evaluate(models={'ReferencePredictor': ReferencePredictor(ref_func='close', name='ReferencePredictor_%s'%today),
    #                  'ModellerPredictor': ModellerPredictor(name='ModellerPredictor_%s'%today)})
    evaluate(models, date=today)
