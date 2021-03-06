# -*- coding: utf-8 -*-
import os

PWD = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.abspath(os.path.join(PWD, '..'))
DATASET_DIR = os.path.join(PACKAGE_ROOT, 'datasets')
TRAINED_MODEL_DIR = os.path.join(PACKAGE_ROOT, 'trained_models')
DATA_FOLDER = os.path.join(DATASET_DIR, 'sign-language-word-dataset')

# MODEL FITTING
VIDEO_SIZE=32
VIDEO_DEPTH=15
BATCH_SIZE=2
EPOCHS=50# 1 for testing, 10 for final model
NUMBER_CLASSES=4
#EPOCHS = int(os.environ.get('EPOCHS', 1))  # 1 for testing, 10 for final model


MODEL_NAME = 'cnn_model'
PIPELINE_NAME = 'cnn_pipe'
CLASSES_NAME = 'classes'
ENCODER_NAME = 'encoder'


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
