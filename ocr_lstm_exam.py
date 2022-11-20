from keras.layers import Input, Dense, Dropout, Activation, LSTM
from keras.layers import Lambda, Convolution2D, MaxPooling2D, Flatten, Reshape
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
from keras.layers.pooling import GlobalAveragePooling1D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import Model
import keras.backend as K

import numpy as np

timesteps = 15;
number_of_samples = 92;
nb_samples = number_of_samples;
frame_row = 32;
frame_col = 32;
channels = 3;2
import glob
import numpy
from PIL import Image
nb_epoch = 11;
batch_size = timesteps;
Path = '/home/nbm/PycharmProjects/english_data/'
#Path2 = '/home/nbm/PycharmProjects/lstm_text'
imagePath = sorted(glob.glob(Path + '/*.bin.png'))
textPath = sorted(glob.glob(Path + '/*.gt.txt'))
im_array = numpy.array( [numpy.array(Image.open(img).resize((frame_row,frame_col)), 'f').flatten() for img in imagePath])

data = numpy.array((im_array, timesteps, frame_row, frame_col, channels))
label = numpy.array((textPath, timesteps, 1))

X_train = data[0:72]
y_train = label[0:72]

X_test = data[72:, ]
y_test = label[72:, ]

# %%

model = Sequential();

model.add(TimeDistributed(Convolution2D(32, 3, 3, border_mode='same'), input_shape=(data-1,timesteps,frame_row,frame_col,channels)))
model.add(TimeDistributed(Activation('relu')))
model.add(TimeDistributed(Convolution2D(32, 3, 3)))
model.add(TimeDistributed(Activation('relu')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(TimeDistributed(Dropout(0.25)))

model.add(TimeDistributed(Flatten()))
model.add(TimeDistributed(Dense(512)))
# output dimension here is (None, 100, 512)

model.add(TimeDistributed(Dense(35, name="first_dense")))
# output dimension here is (None, 100, 35)


model.add(LSTM(output_dim=20, return_sequences=True))
# output dimension here is (None, 100, 20)

time_distributed_merge_layer = Lambda(function=lambda x: K.mean(x, axis=1, keepdims=False))

model.add(time_distributed_merge_layer)
# output dimension here is (None, 1, 20)


# model.add(Flatten())
model.add(Dense(1, activation='sigmoid', input_shape=(None, 20)))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(X_test, y_test))
