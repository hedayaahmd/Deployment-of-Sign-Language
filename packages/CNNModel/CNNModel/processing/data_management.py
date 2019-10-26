# -*- coding: utf-8 -*-
"""
note here when writing this script we don't have model so it would be update
"""


import logging
import pandas as pd
import os
import typing as t
from glob import glob
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

from keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder


from CNNModel import model as m
from CNNModel.config import config


_logger = logging.getLogger(__name__)


def load_single_video(data_folder:str,filename:str)-> pd.DataFrame:
    video_df = []

    # search for specific video in directory
    for video_path in glob(os.path.join(data_folder, f'{filename}')):
        tmp = pd.DataFrame([video_path, 'unknown']).T
        video_df.append(tmp)

    # concatenate the final df
    video_df = pd.concat(video_df, axis=0, ignore_index=True)
    video_df.columns = ['video', 'target']

    return video_df

def load_videos_paths(data_folder):
    """
    Makes dataframe with video path and target
    """
    videos_df = []

  # navigate within each folder
    for class_folder_name in os.listdir(data_folder):
        if class_folder_name == '__init__.py':
            continue
        class_folder_path = os.path.join(data_folder, class_folder_name)

    # collect every video path
        for video_name in os.listdir(class_folder_path):
            video_path=class_folder_path+'/'+video_name
            tmp = pd.DataFrame([video_path, class_folder_name]).T
            videos_df.append(tmp)

  # concatenate the final df
    videos_df = pd.concat(videos_df, axis=0, ignore_index=True)
    videos_df.columns = ['video', 'target']

    return videos_df



def get_train_test_target(df):

	X_train, X_test, y_train, y_test = train_test_split(df['video'],
                                                     df['target'],
                                                     test_size=0.20,
                                                     random_state=101)

	X_train.reset_index(drop=True, inplace=True)
	X_test.reset_index(drop=True, inplace=True)

	y_train.reset_index(drop=True, inplace=True)
	y_test.reset_index(drop=True, inplace=True)

	return X_train, X_test, y_train, y_test


def save_pipeline_keras(model):

    joblib.dump(model.named_steps['dataset'], config.PIPELINE_PATH)
    joblib.dump(model.named_steps['cnn_model'].classes_, config.CLASSES_PATH)
    model.named_steps['cnn_model'].model.save(config.MODEL_PATH)
    remove_old_pipelines(
        files_to_keep=[config.MODEL_FILE_NAME, config.ENCODER_FILE_NAME,
                       config.PIPELINE_FILE_NAME, config.CLASSES_FILE_NAME])


def load_pipeline_keras():
    dataset = joblib.load(config.PIPELINE_PATH)

    build_model = lambda: load_model(config.MODEL_PATH)

    classifier = KerasClassifier(build_fn=build_model,
                          batch_size=config.BATCH_SIZE,
                          validation_split=10,
                          epochs=config.EPOCHS,
                          verbose=2,
                          callbacks=m.callbacks_list,
                          )

    classifier.classes_ = joblib.load(config.CLASSES_PATH)
    classifier.model = build_model()

    return Pipeline([
        ('dataset', dataset),
        ('cnn_model', classifier)
    ])

def load_encoder() -> LabelEncoder:
    encoder = joblib.load(config.ENCODER_PATH)

    return encoder


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines, models, encoders and classes.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ['__init__.py']
    for model_file in Path(config.TRAINED_MODEL_DIR).iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()


if __name__ == '__main__':

    videos_df = load_videos_paths(config.DATA_FOLDER)
    print(videos_df.head())

    TEST_DIR=os.path.join(config.DATASET_DIR,'test_data')
    Test_folder =os.path.join(TEST_DIR,'buy')
    video_1=load_single_video(data_folder=Test_folder,filename='buy_14.avi')
    print(video_1)

    X_train, X_test, y_train, y_test = get_train_test_target(videos_df)
    print(X_train.shape, X_test.shape)
