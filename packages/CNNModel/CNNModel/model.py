# i have error here AttributeError: module 'config' has no attribute 'MODEL_PATH'
#import CNN model libraries
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,Conv3D,MaxPooling3D,BatchNormalization
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier


from CNNModel.config import config

def cnn_model(kernal_size=(3,3,3),
	          pool_size=(3,3,3),
	          first_filter=32,
	          second_filter=32,
	          third_filter=32,
	          dropout_conv=0.5,
	          dropout_dense=0.5,
	          vid_size=32,
	          patch_size=15,
	          channel=1):

	model = Sequential()
	model.add(Conv3D(first_filter,kernal_size, input_shape=(vid_size, vid_size, patch_size,channel), activation='relu'))
	model.add( BatchNormalization())
	model.add(Conv3D(second_filter,kernal_size, input_shape=(vid_size, vid_size, patch_size,channel), activation='relu'))
	model.add( BatchNormalization())
	model.add(MaxPooling3D(pool_size=pool_size))
	model.add( BatchNormalization())

	model.add(Conv3D(third_filter,kernal_size, input_shape=(vid_size, vid_size, patch_size,channel), activation='relu'))
	model.add( BatchNormalization())

	model.add(Dropout(dropout_conv))
	model.add(Flatten())
	model.add(Dense(128, init='normal', activation='relu'))
	model.add(Dropout(dropout_dense))
	model.add(Dense(config.NUMBER_CLASSES,init='normal'))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy', 
				 optimizer='adadelta',
					metrics=['acc'])

	return model


checkpoint = ModelCheckpoint(config.MODEL_PATH,
							monitor='acc',
							verbose=1,
							save_best_only=True,
							mode='max')

reduce_lr = ReduceLROnPlateau(monitor='acc',
							  factor=0.5,
							  patience=2,
                              verbose=1,
                              mode='max',
                              min_lr=0.00001)


callbacks_list = [checkpoint, reduce_lr]


cnn_clf = KerasClassifier(build_fn=cnn_model,
                          batch_size=config.BATCH_SIZE, 
                          validation_split=10,
                          epochs=config.EPOCHS,
                          verbose=2,
                          callbacks=callbacks_list
                          )

if __name__ == '__main__':
    
    model = cnn_model()
    model.summary()

#    import data_management as dm
#    import config
#    import preprocessors as pp
#    
#    model = cnn_model(image_size = config.IMAGE_SIZE)
#    model.summary()
#    
#    images_df = dm.load_image_paths(config.DATA_FOLDER)
#    X_train, X_test, y_train, y_test = dm.get_train_test_target(images_df)
#    
#    enc = pp.TargetEncoder()
#    enc.fit(y_train)
#    y_train = enc.transform(y_train)
#    
#    dataset = pp.CreateDataset(50)
#    X_train = dataset.transform(X_train)
#    
#    cnn_clf.fit(X_train, y_train)