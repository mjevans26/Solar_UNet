# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 12:41:40 2021

@author: MEvans
"""

from scv.utils import model_tools, processing
from scv.utils.prediction_tools import make_pred_dataset, callback_predictions, plot_to_image
from matplotlib import pyplot as plt
import argparse
import os
import glob
import json
import math
import tensorflow as tf
from datetime import datetime
from azureml.core import Run, Workspace, Model

print(tf.__version__)
# Set Global variables

parser = argparse.ArgumentParser()
parser.add_argument('--train_data', type = str, required = True, help = 'Training datasets')
parser.add_argument('--eval_data', type = str, required = True, help = 'Evaluation datasets')
parser.add_argument('--test_data', type = str, default = None, help = 'directory containing test image(s) and mixer')
parser.add_argument('--model_id', type = str, required = False, default = None, help = 'model id for continued training')
parser.add_argument('-lr', '--learning_rate', type = float, default = 0.001, help = 'Initial learning rate')
parser.add_argument('-w', '--weights', type = str, default = None, help = 'sample weight for classes in iou, bce, etc.')
parser.add_argument('--bias', type = float, default = None, help = 'bias value for keras output layer initializer')
parser.add_argument('-e', '--epochs', type = int, default = 10, help = 'Number of epochs to train the model for')
parser.add_argument('-b', '--batch', type = int, default = 16, help = 'Training batch size')
parser.add_argument('--size', type = int, default = 3000, help = 'Size of training dataset')
parser.add_argument('--kernel_size', type = int, default = 256, dest = 'kernel_size', help = 'Size in pixels of incoming patches')
parser.add_argument('--response', type = str, required = True, help = 'Name of the response variable in tfrecords')
parser.add_argument('--bands', type = str, required = False, default = '["B2", "B3", "B4", "B8", "B11", "B12"]')
parser.add_argument('--splits', type = str, required = False, default = '[0]')
parser.add_argument('--epoch_start', type = int, required = False, help = 'If re-training, the last epoch')

args = parser.parse_args()
print('bands', args.bands)
TRAIN_SIZE = args.size
BATCH = args.batch
EPOCHS = args.epochs
LAST = args.epoch_start
BIAS = args.bias
WEIGHTS = json.loads(args.weights)
LR = args.learning_rate
BANDS = json.loads(args.bands)
DEPTH = len(BANDS)
SPLITS = json.loads(args.splits)
if sum(SPLITS) == 0:
    SPLITS = None
RESPONSE = dict({args.response:2})
MOMENTS = [(0,10000),(0,10000),(0,10000),(0,10000),(0,10000),(0,10000) ]
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=LR, beta_1=0.9, beta_2=0.999)

METRICS = {
        'logits':[tf.keras.metrics.CategoricalAccuracy()],
        'classes':[tf.keras.metrics.MeanIoU(num_classes = 2, sparse_y_pred = True, sparse_y_true = False)]
        }

def weighted_crossentropy(y_true, y_pred):
    class_weights = tf.compat.v2.constant(WEIGHTS)
    weights = tf.reduce_sum(class_weights * y_true, axis = -1)
    print('weights shape', weights.shape)
    unweighted_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    weighted_loss = weights * unweighted_loss
    loss = tf.reduce_mean(weighted_loss)
    return loss


LOSSES = {
    'logits':weighted_crossentropy
    }

FEATURES = BANDS + [args.response]

# round the training data size up to nearest 100 to define buffer
BUFFER = math.ceil(args.size/100)*100

# Specify the size and shape of patches expected by the model.
KERNEL_SIZE = args.kernel_size
KERNEL_SHAPE = [KERNEL_SIZE, KERNEL_SIZE]
COLUMNS = [
  tf.io.FixedLenFeature(shape=KERNEL_SHAPE, dtype=tf.float32) for k in FEATURES
]
FEATURES_DICT = dict(zip(FEATURES, COLUMNS))

# create special folders './outputs' and './logs' which automatically get saved
os.makedirs('outputs', exist_ok = True)
os.makedirs('logs', exist_ok = True)
out_dir = './outputs'
log_dir = './logs'

# create training dataset

# train_files = glob.glob(os.path.join(args.data_folder, 'training', 'UNET_256_[A-Z]*.gz'))
# eval_files =  glob.glob(os.path.join(args.data_folder, 'eval', 'UNET_256_[A-Z]*.gz'))

train_files = []
for root, dirs, files in os.walk(args.train_data):
    for f in files:
        train_files.append(os.path.join(root, f))

eval_files = []
for root, dirs, files in os.walk(args.eval_data):
    for f in files:
        eval_files.append(os.path.join(root, f))
        
# train_files = glob.glob(os.path.join(args.train_data, 'UNET_256_[A-Z]*.gz'))
# eval_files =  glob.glob(os.path.join(args.eval_data, 'UNET_256_[A-Z]*.gz'))

training = processing.get_training_dataset(
        files = train_files,
        ftDict = FEATURES_DICT,
        features = BANDS,
        response = RESPONSE,
        moments = MOMENTS,
        buff = BUFFER,
        batch = BATCH,
        axes = [2],
        repeat = True,
        splits = SPLITS)

evaluation = processing.get_eval_dataset(
        files = eval_files,
        ftDict = FEATURES_DICT,
        features = BANDS,
        response = RESPONSE,
        moments = MOMENTS,
        splits = SPLITS)

## DEFINE CALLBACKS

# get the current time
now = datetime.now() 
date = now.strftime("%d%b%y")
date

# define a checkpoint callback to save best models during training
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(out_dir, 'best_weights_'+date+'_{epoch:02d}.hdf5'),
    monitor='val_classes_mean_io_u',
    verbose=1,
    save_best_only=True,
    mode='max'
    )

# define a tensorboard callback to write training logs
tensorboard = tf.keras.callbacks.TensorBoard(log_dir = log_dir)

# get the run context
run = Run.get_context()
exp = run.experiment
ws = exp.workspace

## BUILD THE MODEL

# if a model directory provided we will reload previously trained model and weights
if args.model_id:
    # we will package the 'models' directory within the 'azure' dirrectory submitted with experiment run
    model_dir = Model.get_model_path(args.model_id, _workspace = ws)
#    model_dir = os.path.join('./models', args.model_id, '1', 'outputs')
    
    # load our previously trained model and weights
    model_file = glob.glob(os.path.join(model_dir, '*.h5'))[0]
    weights_file = glob.glob(os.path.join(model_dir, '*.hdf5'))[0]
    m, checkpoint = model_tools.retrain_model(
        model_file = model_file,
        checkpoint = checkpoint,
        eval_data = evaluation,
        metric = 'classes_mean_io_u',
        weights_file = weights_file,
        custom_objects = {'weighted_crossentropy': weighted_crossentropy},
        lr = LR)
    # TODO: make this dynamic
    initial_epoch = LAST
# otherwise build a model from scratch with provided specs
else:
    m = model_tools.get_unet_model(nclasses = 2, nchannels = DEPTH, optim = OPTIMIZER, loss = LOSSES, mets = METRICS, bias = BIAS)
    initial_epoch = 0

# if test images provided, define an image saving callback
if args.test_data:
    
    test_files = glob.glob(os.path.join(args.test_data, '*.gz'))
    mixer_file = glob.glob(os.path.join(args.test_data, '*.json'))
    
    # run predictions on a test image and log so we can see what the model is doing at each epoch
    jsonFile = mixer_file[0]
    with open(jsonFile,) as file:
        mixer = json.load(file)
        
    pred_data = make_pred_dataset(test_files, BANDS, moments = MOMENTS)
    file_writer = tf.summary.create_file_writer(log_dir + '/preds')

    def log_pred_image(epoch, logs):
      out_image = callback_predictions(pred_data, m, mixer)
      prob = out_image[:, :]
      figure = plt.figure(figsize=(10, 10))
      plt.imshow(prob)
      image = plot_to_image(figure)
    
      with file_writer.as_default():
        tf.summary.image("Predicted Image", image, step=epoch)
    
    pred_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end = log_pred_image)
    
    callbacks = [checkpoint, tensorboard, pred_callback]
else:
    callbacks = [checkpoint, tensorboard]
    
# train the model
steps_per_epoch = int(TRAIN_SIZE//BATCH)
m.fit(
        x = training,
        epochs = EPOCHS,
        steps_per_epoch = steps_per_epoch,
        validation_data = evaluation,
        callbacks = callbacks,
        initial_epoch = initial_epoch
        )

m.save(os.path.join(out_dir, f'solar_unet256_{date}.h5'))
