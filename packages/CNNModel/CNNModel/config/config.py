# -*- coding: utf-8 -*-

import os
import pathlib
import CNNModel


PACKAGE_ROOT = pathlib.Path(CNNModel.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'
DATA_DIR = PACKAGE_ROOT / 'datasets'

# MODEL FITTING
VIDEO_SIZE=32
VIDEO_DEPTH=15
BATCH_SIZE=2
EPOCHS=50# 1 for testing, 10 for final model
NUMBER_CLASSES=4
#EPOCHS = int(os.environ.get('EPOCHS', 1))  # 1 for testing, 10 for final model

# MODEL PERSISTING
DATASET_NAME='sign-language-word-dataset'
DATA_FOLDER=os.path.join(DATA_DIR,DATASET_NAME)

MODEL_NAME = 'cnn_model'
PIPELINE_NAME = 'cnn_pipe'
CLASSES_NAME = 'classes'
ENCODER_NAME = 'encoder'
'''
SAVED_WORK_DIR=os.path.join(PACKAGE_ROOT,'trained_models')
MODEL_PATH =os.path.join(SAVED_WORK_DIR, MODEL_NAME)
PIPELINE_PATH = os.path.join(SAVED_WORK_DIR,PIPELINE_NAME)
CLASSES_PATH = os.path.join(SAVED_WORK_DIR,CLASSES_NAME)
ENCODER_PATH = os.path.join(SAVED_WORK_DIR,ENCODER_NAME)'''


with open(os.path.join(PACKAGE_ROOT, 'VERSION')) as version_file:
    _version = version_file.read().strip()

MODEL_FILE_NAME = f'{MODEL_NAME}_{_version}.h5'
MODEL_PATH = os.path.join(TRAINED_MODEL_DIR, MODEL_FILE_NAME)

PIPELINE_FILE_NAME = f'{PIPELINE_NAME}_{_version}.pkl'
PIPELINE_PATH = os.path.join(TRAINED_MODEL_DIR, PIPELINE_FILE_NAME)

CLASSES_FILE_NAME = f'{CLASSES_NAME}_{_version}.pkl'
CLASSES_PATH = os.path.join(TRAINED_MODEL_DIR, CLASSES_FILE_NAME)

ENCODER_FILE_NAME = f'{ENCODER_NAME}_{_version}.pkl'
ENCODER_PATH = os.path.join(TRAINED_MODEL_DIR, ENCODER_FILE_NAME)
