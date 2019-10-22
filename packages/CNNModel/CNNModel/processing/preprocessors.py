# -*- coding: utf-8 -*-

import numpy as np
import cv2
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin

from CNNModel.config import config

class TargetEncoder(BaseEstimator,TransformerMixin):

    def __init__(self,encoder=LabelEncoder()):
        self.encoder=encoder



    def fit(self,X,y=None):
        #X is the target
        self.encoder.fit(X)
        return self

    def transform(self,X):
        X=X.copy()
        X=np_utils.to_categorical(self.encoder.transform(X),config.NUMBER_CLASSES)
        return X



def _vid_resize(df,n,video_size,video_depth):
    frames = []

    cap = cv2.VideoCapture(df[n])
    fps = cap.get(5)
    #print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))

    for k in range(15):
        ret, frame = cap.read()
        frame=cv2.resize(frame,(video_size,video_size),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

                #plt.imshow(gray, cmap = plt.get_cmap('gray'))
                #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
                #plt.show()
                #cv2.imshow('frame',gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    input=np.array(frames)

    #print(input.shape)
    ipt=np.rollaxis(np.rollaxis(input,2,0),2,0)
    #print(ipt.shape)
    return ipt



class CreateDataset(BaseEstimator,TransformerMixin):

    def __init__(self,video_size=32,video_depth=15):
        self.video_size=video_size
        self.video_depth=video_depth

    def fit(self,X,y=None):
        return self

    def transform(self,x):
        if x.shape == (1,2):
            x=x['video']
            x.reset_index(drop=True, inplace=True)
        X=x.to_frame()
        tmp = np.zeros((len(X),self.video_size,self.video_size, self.video_depth),dtype='float32')

        for n in range(0, len(X)):
            vid = _vid_resize(X['video'], n,self.video_size,self.video_depth)
            tmp[n] = vid
        reshaped_tmp=tmp.reshape(tmp.shape[0],tmp.shape[1],tmp.shape[2],tmp.shape[3],1)
        return reshaped_tmp


if __name__ == '__main__':

    import CNNModel.processing.data_management as dm
    from CNNModel.config import config
    import os

    videos_df =dm.load_videos_paths(config.DATA_FOLDER)
    X_train, X_test, y_train, y_test = dm.get_train_test_target(videos_df)

    enc = TargetEncoder()
    enc.fit(y_train)
    y_train = enc.transform(y_train)
    #print(y_train)
    dataCreator = CreateDataset()
    print(X_train.head())
    print(type(X_train))
    X_train = dataCreator.transform(X_train)
    print(X_train.shape)
    print(type(X_train))



    print("for testing one video loading")
    TEST_DIR=os.path.join(config.DATA_DIR,'test_data')
    Test_folder =os.path.join(TEST_DIR,'buy')
    video_1=dm.load_single_video(data_folder=Test_folder,filename='buy_14.avi')
    P_try=dataCreator.transform(video_1)
    print(type(P_try))
    print(P_try.shape)
