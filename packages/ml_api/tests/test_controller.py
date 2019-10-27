import io
import json
import math
import os

from CNNModel.config import config as ccn_config

from api import __version__ as api_version
from CNNModel import __version__ as _version

def test_sign_endpoint_returns_200(flask_test_app):
    #ok

    response=flask_test_app.get('/sign')

    assert response.status_code ==200



def test_version_endpoint_returns_version(flask_test_app):
    # When
    response = flask_test_app.get('/version')
    # Then
    assert response.status_code == 200
    response_json = json.loads(response.data)
    assert response_json['model_version'] == _version
    assert response_json['api_version'] == api_version



def test_classifier_endpoint_returns_prediction(flask_test_app):
    pass
    data_dir = os.path.abspath(os.path.join(ccn_config.DATA_FOLDER, os.pardir))
    test_dir = os.path.join(data_dir, 'test_data')
    buy_dir = os.path.join(test_dir, 'buy')
    buy_vid = os.path.join(buy_dir, 'buy_14.avi')
    with open(buy_vid, "rb") as vid_file:
        file_bytes = vid_file.read()
        data = dict(
            file=(io.BytesIO(bytearray(file_bytes)), "buy_14.avi"),
        )

    # When
    response = flask_test_app.post('/predict/classifier',
                                      content_type='multipart/form-data',
                                      data=data)

    # Then
    assert response.status_code == 200
    response_json = json.loads(response.data)
    assert response_json['readable_predictions']
