import os
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint

from ssd.utils.utils import print_box, get_args_training, get_model_by_name

from ssd.models.ssd_model import *
from ssd.training_manager import TrainingManager
from ssd.loss import FocalLoss

if __name__ == "__main__":

    nbrepoch, batchsize, datadir, model, weights = get_args_training()

    """Setup dirs and names"""
    type = "train" if weights != "None" else "transf"

    train_name = "{}_{}_{}".format(model, type, datadir[0].split('/')[-1])
    directory = "./graph/{}".format(train_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    train = TrainingManager(train_name, batch_size=batchsize)
    info = "{} in => {}".format(train_name, directory)
    print_box(info, len(info) + 4)

    """Setup of model for training"""
    model = get_model_by_name(model)
    train.setModel(model, weightpath=weights)
    train.setLoss(FocalLoss())
    train.setOptimizer(Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=0.001, decay=0.0))

    train.setInputGenerator(datadir)

    """Setup of callbacks functions"""
    tbd = TensorBoard(
        log_dir=directory, histogram_freq=0,
        write_graph=True, write_images=True
    )
    train.setCallbacks(tbd)
    mcp = ModelCheckpoint(
        directory + '/weights.{epoch:03d}.h5',
        verbose=1, save_weights_only=True,
    )
    train.setCallbacks(mcp)

    """Run training"""
    train.compile()
    train.train(epochs=nbrepoch)

    """Exit prog"""
    exit(0)
