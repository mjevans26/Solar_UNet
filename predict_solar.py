# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 19:51:32 2021

@author: MEvans
"""

from utils import model_tools, processing
from utils.prediction_tools import makePredDataset, write_tfrecord_predictions
from matplotlib import pyplot as plt
import argparse
import os
import glob
import json
import math
import tensorflow as tf
from datetime import datetime
from azureml.core import Run, Workspace, Model, Datastore, Dataset
from azure.storage.blob import BlobClient


# Set Global variables

parser = argparse.ArgumentParser()

# parser.add_argument('--pred_data', type = str, default = True, help = 'directory containing test image(s) and mixer')
# parser.add_argument('--model_id', type = str, required = True, default = None, help = 'model id for continued training')

parser.add_argument('--kernel_size', type = int, default = 256, dest = 'kernel_size', help = 'Size in pixels of incoming patches')
parser.add_argument('--bands', type = str, nargs = '+', required = False, default = '["B2", "B3", "B4", "B8", "B11", "B12"]')
parser.add_argument('-c', type=str, help='The path to the job config file')
parser.add_argument('--aoi', type=str, required = True, default = 'Delaware', help = 'Name of the area to run predictions')
parser.add_argument('--year', type=str, required = True, default = 'Summer2020', help = 'Season and year subdirectory')

args = parser.parse_args()

# # get the run context
# run = Run.get_context()
# exp = run.experiment
# read annual config file
with open(args.c, 'r') as f:
    config = json.load(f)

# access relevant key values
blob = config['blobContainer']
wksp = config['workspace']
model = config['model']

# load workspace configuration from the config.json file in the current folder.
ws = Workspace(subscription_id = wksp["subscription_id"], workspace_name = wksp["workspace_name"], resource_group = wksp["resource_group"])

# access our registered data share containing image data in this workspace
datastore = Datastore.get(workspace = ws, datastore_name = blob['datastore_name'])
pred_path = (datastore, config['data'].format(args.aoi, args.year))
# pred_path = (datastore, 'CPK_solar/data/predict/testpred6')
blob_files = Dataset.File.from_files(path = [pred_path])

# BANDS = args.bands
BANDS = json.loads(args.bands)
OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999)

METRICS = {
        'logits':[tf.keras.metrics.MeanSquaredError(name='mse'), tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')],
        'classes':[tf.keras.metrics.MeanIoU(num_classes=2, name = 'mean_iou')]
        }

def get_weighted_bce(y_true, y_pred):
    return model_tools.weighted_bce(y_true, y_pred, 1)

print(f'Loading model {config["model"]}')
# if a model directory provided we will reload previously trained model and weights
# we will package the 'models' directory within the 'azure' dirrectory submitted with experiment run
model_dir = Model.get_model_path(model, _workspace = ws)
#    model_dir = os.path.join('./models', args.model_id, '1', 'outputs')

# load our previously trained model and weights
model_file = glob.glob(os.path.join(model_dir, '*.h5'))[0]
weights_file = glob.glob(os.path.join(model_dir, '*.hdf5'))[0]
m = model_tools.get_binary_model(depth = len(BANDS), optim = OPTIMIZER, loss = get_weighted_bce, mets = METRICS, bias = None)
m.load_weights(weights_file)

print('found model file:', model_file, '/n weights file:', weights_file)

# Specify the size and shape of patches expected by the model.
KERNEL_SIZE = args.kernel_size
KERNEL_SHAPE = [KERNEL_SIZE, KERNEL_SIZE]


# create special folders './outputs' and './logs' which automatically get saved
os.makedirs('outputs', exist_ok = True)
os.makedirs('logs', exist_ok = True)
out_dir = './outputs'
log_dir = './logs'

testFiles = []

print('loading prediction data')
with blob_files.mount() as mount:
    mount_point = mount.mount_point
    for root, dirs, files in os.walk(mount_point):
        for f in files:
            testFiles.append(os.path.join(root, f))

    predFiles = [x for x in testFiles if '.gz' in x]
    jsonFiles = [x for x in testFiles if '.json' in x]
    jsonFile = jsonFiles[0]
    predData = makePredDataset(predFiles, BANDS, one_hot = None)

    predictions = m.predict(predData, steps=None, verbose=1)

base = os.path.basename(jsonFile)[:-10]
write_tfrecord_predictions(
    predictions = predictions,
    pred_path = out_dir, 
    # pred_path = '.',
    # out_image_base = 'raw_unet256_testpred_solar_Jun21',
    out_image_base = f'{base}_{model}', 
    kernel_shape = KERNEL_SHAPE,
    kernel_buffer = [128,128])

# get the current time
now = datetime.now() 
date = now.strftime("%d%b%y")
date

print('moving predicitons to blob')
# blob_url = "https://aiprojects.blob.core.windows.net/solar/CPK_solar/data/predict/Delaware/outputs/tfrecord/testpred.tfrecords?sp=racw&st=2022-01-27T18:38:10Z&se=2022-01-29T02:38:10Z&sv=2020-08-04&sr=c&sig=vrHeB7LHAc2R2B6rhS%2BwRLqYM4xY5v1%2B9SlGyj8TTIY%3D"
blob_url = blob['blob_url'].format(args.aoi, args.year)
blob_client = BlobClient.from_blob_url(blob_url)
# with open(f'./raw_unet256_testpred_solar_Jun21.tfrecords', 'rb') as f:
with open(f'{out_dir}/{base}_{model}.tfrecords', 'rb') as f:
    blob_client.upload_blob(f)


