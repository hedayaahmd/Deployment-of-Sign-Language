from CNNModel import __version__ as _version
from CNNModel.predict import (make_single_prediction)


def test_make_prediction_on_sample(angry_dir):
    # Given
    filename = 'angry_14.avi'
    expected_classification = 'angry'

    # When
    results = make_single_prediction(video_name=filename,
                                    video_directory=angry_dir
                                     )
    print(results)
    # Then
    assert results['predictions'] is not None
    assert results['readable_predictions'][0] == expected_classification
    assert results['version'] == _version
