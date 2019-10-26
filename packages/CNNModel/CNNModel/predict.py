# -*- coding: utf-8 -*-


import logging

import pandas as pd

from CNNModel import __version__ as _version
from CNNModel.processing import data_management as dm

_logger = logging.getLogger(__name__)
KERAS_PIPELINE = dm.load_pipeline_keras()
ENCODER = dm.load_encoder()

def make_single_prediction(*, video_name: str, video_directory: str):
    """Make a single prediction using the saved model pipeline.
        Args:
            video_name: Filename of the video to classify
            video_directory: Location of the video to classify
        Returns
            Dictionary with both raw predictions and readable values.
        """

    video_df = dm.load_single_video(
        data_folder=video_directory,
        filename=video_name)

    prepared_df = video_df['video'].reset_index(drop=True)
    _logger.info(f'received input array: {prepared_df}, '
                 f'filename: {video_name}')

    predictions = KERAS_PIPELINE.predict(prepared_df)
    readable_predictions = ENCODER.encoder.inverse_transform(predictions)

    _logger.info(f'Made prediction: {predictions}'
                 f' with model version: {_version}')

    return dict(predictions=predictions,
                readable_predictions=readable_predictions,
                version=_version)


def make_bulk_prediction(*, videos_df: pd.Series) -> dict:
    """Make multiple predictions using the saved model pipeline.
    Currently, this function is primarily for testing purposes,
    allowing us to pass in a directory of videos for running
    bulk predictions.
    Args:
        video_df: Pandas series of video
    Returns
        Dictionary with both raw predictions and their classifications.
    """

    _logger.info(f'received input df: {videos_df}')

    predictions = KERAS_PIPELINE.predict(videos_df)
    readable_predictions = ENCODER.encoder.inverse_transform(predictions)

    _logger.info(f'Made predictions: {predictions}'
                 f' with model version: {_version}')

    return dict(predictions=predictions,
                readable_predictions=readable_predictions,
                version=_version)





if __name__ == '__main__':
    import os
    from CNNModel.config import config

    '''
    for making multiple pridection
    '''
    #video_df=dm.load_videos_paths(config.DATA_FOLDER)

    #results=make_bulk_prediction(videos_df=video_df['video'])
    #print(results)


    '''
    single prediction
    '''
    filename = 'angry_14.avi'
    expected_classification = 'angry'
    test_data_dir = os.path.join(config.DATASET_DIR, 'test_data')
    angry_dir = os.path.join(test_data_dir, 'angry')

    # When
    results = make_single_prediction(video_name=filename,
                                    video_directory=angry_dir
                                     )
    print(results)
