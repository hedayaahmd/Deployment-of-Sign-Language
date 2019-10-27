from flask import Blueprint,request,jsonify
from CNNModel.predict import make_single_prediction

from werkzeug.utils import secure_filename

from CNNModel import __version__ as _version
from api import __version__ as api_version
from api.validation import allowed_file

import os

from api.config import get_logger ,UPLOAD_FOLDER
#from api

_logger=get_logger(logger_name=__name__)

prediction_app=Blueprint('prediction_app',__name__)



@prediction_app.route('/version',methods=['GET'])
def version():
    ''' this for making versioning requst'''

    if request.method == 'GET':
        return jsonify({'model_version':_version,
                        'api_version':api_version})



@prediction_app.route('/sign',methods=['GET'])
def sign():
    if request.method == 'GET':
        _logger.info('sign status ok')
        return 'ok hedaya'

@prediction_app.route('/predict/classifier',methods=['POST'])
def predict_video():
    if request.method == 'POST':
        #check if the post request has the file part
        if 'file' not in request.files:
            return jsonify('no file found') , 400
        file=request.files['file']

        #Basic file extension validation

        if file and allowed_file(file.filename):
            filename=secure_filename(file.filename)

        # save the file to upload directory
        file.save(os.path.join(UPLOAD_FOLDER,filename))

        _logger.debug(f'inputs: {filename}')

        #perform predction

        result =make_single_prediction(video_name=filename,
                                        video_directory=UPLOAD_FOLDER)
        _logger.debug(f'Outputs: {result}')

        readable_predictions = result.get('readable_predictions')
        version = result.get('version')
        # return the output results
        return jsonify(
            {'readable_predictions': readable_predictions[0],
             'version': version})
