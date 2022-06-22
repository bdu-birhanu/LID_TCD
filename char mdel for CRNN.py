from keras import backend as K
from keras.models import Sequential,Model,load_model
from keras.utils import plot_model
from  keras.utils import  np_utils
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU
from sklearn.metrics import classification_report, confusion_matrix
#from keras.layers MaxPooling2D
from keras.layers import Input
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers.convolutional import Conv2D,MaxPooling2D
from sklearn.utils import shufflel
import shutil
import numpy as np
from sklearn.model_selection import train_test_split #insteade of crossvalidation we use modele_selection for this version
from PIL import Image
import matplotlib.pyplot as plt
import numpy
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import cv2
from keras.models import Sequential
import re
import numpy
import np_utils
from keras.utils import plot_model
from  keras.utils import  np_utils
import  tensorflow as tf
from IPython.display import SVG,Image,display
from keras.utils.vis_utils import model_to_dot

from keras.layers.core import Dense, Dropout, Activation, Flatten
from sklearn.metrics import classification_report, confusion_matrix
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam
from keras.layers.convolutional import Conv2D
'''from keras import backend as bk
import numpy as np
import os
import Tkinter
from PIL import Image,ImageDraw, ImageFont
from numpy import *'''
# SKLEARN
from sklearn.utils import shuffle
import shutil
from sklearn.model_selection import train_test_split #insteade of crossvalidation we use modele_selection for this version
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import glob
# input image dimensions
im_rows, im_cols = 96,64
# number of channels
img_channels = 3


def im_resize(myimage):
  (wt,ht)=(im_rows,im_cols)
  (h,w)=myimage.shape
  fw=float(w)/wt
  fh=float(h)/ht
  f=max(fw,fh)
  newsize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1))
  myimage = cv2.resize(myimage, newsize)
  target = np.ones([ht, wt]) * 255
  target[0:newsize[1], 0:newsize[0]] = myimage
  #newimage = target
  newimage = cv2.transpose(target)
  (m, s) = cv2.meanStdDev(newimage)
  m=[0][0]
  s=s[0][0]
  newimage=newimage-m
  newimage=newimage/s if s>0 else newimage
  return newimage

Path ='/home/belay/PycharmProjects/2nd_round/image_char_78'
imagepath = sorted(glob.glob(Path + '/*.bin.png'))
text =open('/home/belay/PycharmProjects/2nd_round/image_char_78/char.txt','r').read().decode('utf-8')
lines = text.split('\n')
chars = sorted(list(set(text)))
mapping = dict((c, i) for i, c in enumerate(chars))  # map charcter to index
mapping1 = dict((i, c) for i, c in enumerate(chars))
sequences = list()
for line in lines:
    # integer encode line
    if len(line)==0: continue
    encoded_seq = [mapping[char] for char in line]
    # store
    sequences.append(encoded_seq)
label = [list(line) for line in sequences]
lable=np.array(label)


image = [np.array(Image.open(i), 'f') for i in imagepath]
img = []
for i in image:
    #imgr=cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)# it is not necessay sinmce it is gray scale image synthericaly generated
    #im1=cv2.resize(i,(img_col,img_row),interpolation=cv2.INTER_AREA)
    #imgt=im1/255
    img.append(im_resize(i))
    #img[1].shape  [500,48]
#im = sequence.pad_sequences(img, value=float(255), dtype='float32',
                            #padding="post", truncating='post')
im=np.array(img)
data, labels = shuffle(im_array, label, random_state=2)
training=[data,labels]
print training
print training[0]
print training[1]
(x_train,y_train)=(training[0],training[1])
print len(x_train),len(y_train)
#print label[2999]
# batch_size to train
batch_size =100
# number of output classes
nb_clas = 284
# number of epochs to train
nb_epoch =10

#print label[2515:2520]
#print label[395:465]
# STEP 1: split X and y into training and testing sets
im=np.load('/home/belay/PycharmProjects/2nd_round/image_char_78/chars_im.npy')
tex=np.load('/home/belay/PycharmProjects/2nd_round/image_char_78/chars_tx.npy')



X_train, X_test, y_train, y_test = train_test_split(im, tex, test_size=0.1, random_state=4)
print sorted(y_test)

#words, we want to transform our dataset from having shape (n, width, height) to (n, depth, width, height)..this is for theano backend
#but for tensorflow backedn we will have(n,width,height,depth)
#if bk.image_data_format() == 'channels_first':
X_train = X_train.reshape(X_train.shape[0],64,96,1)
X_test = X_test.reshape(X_test.shape[0], 64,96,1)

input_shape = (64,96,1)
#convert our data type to float32
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#normalize our data values to the range [0, 1].
X_train /= 255
X_test /= 255
# Convert 1-dimensional class arrays to number of class-dimensional class matrices(one hot encoding
Y_train =np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
print Y_test
print 'number of sample vs class',Y_train.shape
print 'number of sample vs class',Y_test.shape

# to print the number of image with their shape in training/testing datset
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3
act='relu'
input_data=Input(shape=input_shape, dtype='float32')
inner=Conv2D(nb_filters, nb_conv,padding="same",
                         activation=act, kernel_initializer='he_normal', name='conv1')(input_data)
inner=Conv2D(nb_filters, nb_conv,padding="same",
                         activation=act, kernel_initializer='he_normal', name='conv2')(inner)
inner =MaxPooling2D(pool_size=(nb_pool, nb_pool))(inner)
#model.add(Dropout(rate=0.25))
inner=Conv2D(nb_filters, nb_conv,padding="same",
                         activation=act, kernel_initializer='he_normal', name='conv3')(inner)
inner=Conv2D(nb_filters, nb_conv,padding="same",
                         activation=act, kernel_initializer='he_normal', name='conv4')(inner)
inner =MaxPooling2D(pool_size=(nb_pool, nb_pool))(inner)

inner=Conv2D(nb_filters, nb_conv,padding="same",
                         activation=act, kernel_initializer='he_normal', name='conv5')(inner)
inner=Conv2D(64, nb_conv,padding="same",
                         activation=act, kernel_initializer='he_normal', name='conv6')(inner)

inner =MaxPooling2D(pool_size=(nb_pool, nb_pool))(inner)


inner =Flatten()(inner)
inner =Dense(512)(inner)
inner =Activation('relu')(inner)
inner =Dropout(rate=0.25)(inner)
inner =Dense(512)(inner)
inner =Activation('relu')(inner)
inner =Dropout(rate=0.5)(inner)
out_r =Dense(nb_clas,activation="softmax")(inner)


model=Model(inputs=input_data, outputs=out_r)



model.compile(loss='categorical_crossentropy', optimizer='ADAM', metrics=['accuracy'])


hist = model.fit(X_train,y_train, batch_size=100, epochs=15,verbose=1, validation_split=0.1,shuffle=False)

model_new.save_weights("model_cnn_for_CRNN.h5")




print("Loaded model from disk")'''
#testing model so as to predict the class using testset with out class.
classs=model.predict_classes(X_test)# dispaly the  predicted class
#print  y_test
#print sorted(classs)
#print classs
score = model.evaluate(X_test,Y_test,batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

''''
from sklearn.metrics import accuracy_score
accuracy_score(y_test,classs)'''

# visualizing  accuracy per epoch
train_acc = hist.history['acc']
val_acc = hist.history['val_acc']
epochs = range(nb_epoch)

# to print training  accuracy vs validation accuracy
plt.plot(epochs, train_acc)
plt.plot(epochs, val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train', 'val'], loc=4)
plt.show()

# to display and save model architecture to the spacifed path the

plot_model(model,show_shapes=True, to_file='/home/belay/PycharmProjects/untitled/ocr/amahric_model.png')




