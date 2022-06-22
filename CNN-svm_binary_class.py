from keras import backend as K
from keras.models import Sequential,Model
from keras.utils import plot_model
from keras.utils import  np_utils
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU
from sklearn.metrics import classification_report, confusion_matrix
#from keras.layers MaxPooling2D
from keras.layers import Input
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers.convolutional import Conv2D,MaxPooling2D
from sklearn.utils import shuffle
import shutil
import numpy as np
from sklearn.model_selection import train_test_split 
from PIL import Image
import matplotlib.pyplot as plt
import numpy
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import cv2
#====================variables and network_paramenter=======================================
# input image dimensions
img_rows, img_cols = 128,64
img_channels = 3 #number of channel
batch_size =100
nb_classes = 2# number of output classes
nb_epoch =10 # number of epochs to train
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

Path = '/home/belay/PycharmProjects/2nd_round/hw_printed_data/*/*/'
#Path2 = '/home/belay/PycharmProjects/ocropy-master/tests/output/label_80/'
imagePath = sorted(glob.glob(Path + '/*.bin.png'))
text =open('/home/belay/PycharmProjects/2nd_round/hw_printed_data/hwplabel.txt','r').read().split('\n')

charlable=[]#to store each character as a list to compute the accuracy
for i in text:
    if len(i)==0: continue
    charlable.append(i)# we can use either append (i)--to returen unicode value or .append(y[i])--returen integer value

label=charlable
print len(label)
#convert image to gray resize and then flaten the size of the image in to 1*row or by columan
im_array = numpy.array( [numpy.array(Image.open(img).resize((img_rows,img_cols)), 'f').flatten() for img in imagePath])
print len(im_array)
training=[im_array,label]

(x_train,y_train)=(training[0],training[1])
# print len(x_train),len(y_train)

X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)#, random_state=4)
print (y_test)

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols,1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols,1)

input_shape = (img_rows, img_cols,1)
#convert our data type to float32
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#normalize our data values to the range [0, 1].
X_train /= 255
X_test /= 255
# Convert 1-dimensional class arrays to number of class-dimensional class matrices(one hot encoding
Y_train =np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

input_data = Input(shape=input_shape, dtype='float32')
#handwritten_model = Sequential()
act='relu'
inner = Conv2D(nb_filters, nb_conv, padding='same',
           activation=act, kernel_initializer='he_normal',
           name='conv1')(input_data)
inner =Conv2D(nb_filters, nb_conv,
                border_mode="same",
                input_shape=input_shape,
                activation="relu")(inner)
inner =MaxPooling2D(pool_size=(nb_pool, nb_pool))(inner)
#inner =Dropout(rate=0.2))
inner =Conv2D(nb_filters, nb_conv,border_mode="same",
         activation="relu")(inner)
inner =MaxPooling2D(pool_size=(nb_pool, nb_pool))(inner)
#inner =Dropout(rate=0.2))
inner =Conv2D(nb_filters, nb_conv,border_mode="same",
         activation="relu")(inner)
inner =Conv2D(nb_filters, nb_conv, nb_conv,border_mode="same",
         activation="relu")(inner)
inner =MaxPooling2D(pool_size=(nb_pool, nb_pool))(inner)
inner =Dropout(rate=0.2)(inner)

inner =Conv2D(nb_filters, nb_conv, border_mode="same",
                        activation="relu")(inner)
#
inner =MaxPooling2D(pool_size=(nb_pool, nb_pool))(inner)
#  inner =Dropout(rate=0.25))
inner =Conv2D(128, nb_conv, border_mode="same",activation="relu")(inner)
inner =Conv2D(64, nb_conv, border_mode="same",activation="relu")(inner)
inner =MaxPooling2D(pool_size=(nb_pool, nb_pool))(inner)
inner =Dropout(rate=0.25)(inner)

inner =Flatten()(inner)
inner =Dense(256)(inner)
inner =Activation('relu')(inner)
inner =Dense(256)(inner)
inner =Activation('relu')(inner)
inner =Dropout(rate=0.1)(inner)
out =Dense(nb_classes,activation="softmax")(inner)


model_new = Model(input_data, out)
model_new.summary()
model_new.compile(optimizer = Adam(lr=.00025) , loss = 'categorical_crossentropy', metrics=['accuracy'])
#Training
hist=model_new.fit(X_train, y_train, epochs=20,validation_split=0.2)
 #testing
y=model_new.predict(X_test)
y_pred_class = np.argmax(y_pred,axis=1) # convert predicted labels into classes: say [0.00023, 0.923, 0.031] -->  [1] i.e. negative

score = model_new.evaluate(x_val,y_val)
print('Accuracy on Validation Set',score[1])

feature = Model(inputs=model_new.input,outputs=model_new.get_layer('dense_1').output)
#==================if we want to load the model+++=========
 model_newb=load_model('/home/belay/PycharmProjects/2nd_round/results/discrimination_handwritten_printed/disccrimination.h5')
 model_newb.summary()
#=================================================

x_train_new = feature.predict(X_train)
print(x_train_new.shape)

x_val_new = feature.predict(x_val)
print(x_val_new.shape)

x_test_new= feature.predict(X_test)
print(x_test.shape)
from sklearn.svm import SVC

svm = SVC(kernel='rbf')

svm.fit(x_train_new,np.argmax(Y_train,axis=1))

print('fitting done !!!')

svm.score(x_train_new,np.argmax(Y_train,axis=1))
svm.score(feat_val,np.argmax(y_val,axis=1))
Pred_labels = svm.predict(x_test_new)
#train_with searched parameter
from sklearn.model_selection import GridSearchCV
parameters={'kernel':['rbf'],'C':[1,10,100,100],'gamma':[1e-3,1e-4]}
cls=GridSearchCV(SVC(),parameters)
cls.fit(x_train_new,np.argmax(Y_train,axis=1))
#select the best estimator
svmcls=cls.best_estimator_
svmcls.fit(x_train_new,np.argmax(Y_train,axis=1))
Pred_labels_search=svmcls.pridict(x_test_new)
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score
print classification_report(np.argmax(Y_test,axis=1),svm)# the argmax here returns the catgorical value from  onehot encoding
 print("Accuracy:{0}".format(accuracy_score(np.argmax(Y_test,axis=1),Pred_labels)))
