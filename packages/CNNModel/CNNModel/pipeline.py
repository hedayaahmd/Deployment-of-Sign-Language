# -*- coding: utf-8 -*-

from sklearn.pipeline import Pipeline

from CNNModel.config import config
from CNNModel.processing import preprocessors as pp
from CNNModel import model

pipe = Pipeline([
                ('dataset', pp.CreateDataset(config.VIDEO_SIZE,config.VIDEO_DEPTH)),
                ('cnn_model', model.cnn_clf)
            ])



















if __name__ == '__main__':

    from sklearn.metrics import  accuracy_score
    from CNNModel.processing import data_management as dm
    from CNNModel.config import config

    images_df = dm.load_videos_paths(config.DATA_FOLDER)
    X_train, X_test, y_train, y_test = dm.get_train_test_target(images_df)

    enc = pp.TargetEncoder()
    enc.fit(y_train)
    y_train = enc.transform(y_train)

    pipe.fit(X_train, y_train)

    test_y = enc.transform(y_test)
    predictions = pipe.predict(X_test)

    acc = accuracy_score(enc.encoder.transform(y_test),
                   predictions,
                   normalize=True,
                   sample_weight=None)

    print('Acuracy: ', acc)
