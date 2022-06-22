
from keras.models import Sequential,Model,load_model
import re
import np_utils
from keras.utils import plot_model
from  keras.utils import  np_utils
import  tensorflow as tf
from IPython.display import SVG,Image,display
from keras.utils.vis_utils import model_to_dot
from keras.layers import Input
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
img_rows, img_cols = 32,32
# number of channels
img_channels = 3
r0=[0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98, 105, 112, 119, 126, 133, 140, 147, 154, 161, 168, 175, 182, 189, 196, 203, 210, 217, 224, 1, 8, 15, 22, 29, 36, 43, 50, 57, 64, 71, 78, 85, 92, 99, 106, 113, 120, 127, 134, 141, 148, 155, 162, 169, 176, 183, 190, 197, 204, 211, 218, 225, 2, 9, 16, 23, 30, 37, 44, 51, 58, 65, 72, 79, 86, 93, 100, 107, 114, 121, 128, 135, 142, 149, 156, 163, 170, 177, 184, 191, 198, 205, 212, 219, 226, 3, 10, 17, 24, 31, 38, 45, 52, 59, 66, 73, 80, 87, 94, 101, 108, 115, 122, 129, 136, 143, 150, 157, 164, 171, 178, 185, 192, 199, 206, 213, 220, 227, 4, 11, 18, 25, 32, 39, 46, 53, 60, 67, 74, 81, 88, 95, 102, 109, 116, 123, 130, 137, 144, 151, 158, 165, 172, 179, 186, 193, 200, 207, 214, 221, 228, 5, 12, 19, 26, 33, 40, 47, 54, 61, 68, 75, 82, 89, 96, 103, 110, 117, 124, 131, 138, 145, 152, 159, 166, 173, 180, 187, 194, 201, 208, 215, 222, 229, 6, 13, 20, 27, 34, 41, 48, 55, 62, 69, 76, 83, 90, 97, 104, 111, 118, 125, 132, 139, 146, 153, 160, 167, 174, 181, 188, 195, 202, 209, 216, 223, 230]
charsr =[u'\u1200', u'\u1201', u'\u1202', u'\u1203', u'\u1204', u'\u1205', u'\u1206', u'\u1208', u'\u1209', u'\u120a', u'\u120b', u'\u120c', u'\u120d', u'\u120e', u'\u1210', u'\u1211', u'\u1212', u'\u1213', u'\u1214', u'\u1215', u'\u1216', u'\u1218', u'\u1219', u'\u121a', u'\u121b', u'\u121c', u'\u121d', u'\u121e', u'\u1220', u'\u1221', u'\u1222', u'\u1223', u'\u1224', u'\u1225', u'\u1226', u'\u1228', u'\u1229', u'\u122a', u'\u122b', u'\u122c', u'\u122d', u'\u122e', u'\u1230', u'\u1231', u'\u1232', u'\u1233', u'\u1234', u'\u1235', u'\u1236', u'\u1238', u'\u1239', u'\u123a', u'\u123b', u'\u123c', u'\u123d', u'\u123e', u'\u1240', u'\u1241', u'\u1242', u'\u1243', u'\u1244', u'\u1245', u'\u1246', u'\u1260', u'\u1261', u'\u1262', u'\u1263', u'\u1264', u'\u1265', u'\u1266', u'\u1270', u'\u1271', u'\u1272', u'\u1273', u'\u1274', u'\u1275', u'\u1276', u'\u1278', u'\u1279', u'\u127a', u'\u127b', u'\u127c', u'\u127d', u'\u127e', u'\u1280', u'\u1281', u'\u1282', u'\u1283', u'\u1284', u'\u1285', u'\u1286', u'\u1290', u'\u1291', u'\u1292', u'\u1293', u'\u1294', u'\u1295', u'\u1296', u'\u1298', u'\u1299', u'\u129a', u'\u129b', u'\u129c', u'\u129d', u'\u129e', u'\u12a0', u'\u12a1', u'\u12a2', u'\u12a3', u'\u12a4', u'\u12a5', u'\u12a6', u'\u12a8', u'\u12a9', u'\u12aa', u'\u12ab', u'\u12ac', u'\u12ad', u'\u12ae', u'\u12b8', u'\u12b9', u'\u12ba', u'\u12bb', u'\u12bc', u'\u12bd', u'\u12be', u'\u12c8', u'\u12c9', u'\u12ca', u'\u12cb', u'\u12cc', u'\u12cd', u'\u12ce', u'\u12d0', u'\u12d1', u'\u12d2', u'\u12d3', u'\u12d4', u'\u12d5', u'\u12d6', u'\u12d8', u'\u12d9', u'\u12da', u'\u12db', u'\u12dc', u'\u12dd', u'\u12de', u'\u12e0', u'\u12e1', u'\u12e2', u'\u12e3', u'\u12e4', u'\u12e5', u'\u12e6', u'\u12e8', u'\u12e9', u'\u12ea', u'\u12eb', u'\u12ec', u'\u12ed', u'\u12ee', u'\u12f0', u'\u12f1', u'\u12f2', u'\u12f3', u'\u12f4', u'\u12f5', u'\u12f6', u'\u1300', u'\u1301', u'\u1302', u'\u1303', u'\u1304', u'\u1305', u'\u1306', u'\u1308', u'\u1309', u'\u130a', u'\u130b', u'\u130c', u'\u130d', u'\u130e', u'\u1320', u'\u1321', u'\u1322', u'\u1323', u'\u1324', u'\u1325', u'\u1326', u'\u1328', u'\u1329', u'\u132a', u'\u132b', u'\u132c', u'\u132d', u'\u132e', u'\u1330', u'\u1331', u'\u1332', u'\u1333', u'\u1334', u'\u1335', u'\u1336', u'\u1338', u'\u1339', u'\u133a', u'\u133b', u'\u133c', u'\u133d', u'\u133e', u'\u1340', u'\u1341', u'\u1342', u'\u1343', u'\u1344', u'\u1345', u'\u1346', u'\u1348', u'\u1349', u'\u134a', u'\u134b', u'\u134c', u'\u134d', u'\u134e', u'\u1350', u'\u1351', u'\u1352', u'\u1353', u'\u1354', u'\u1355', u'\u1356']
digit=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230]

y=dict(zip(charsr,digit))
special = open("/home/belay/PycharmProjects/ocropy-master/tests/output/g.txt", "r").read().decode('utf-8')
#print len(special)
Path = '/home/belay/PycharmProjects/ocropy-master/tests/output/image_80/'
imagePath = sorted(glob.glob(Path + '/*.png'))
text =open('/home/belay/PycharmProjects/ocropy-master/tests/output/xl.txt','r').read().decode('utf-8').split('\n')
'''
images = [cv2.imread(file) for file in sorted(glob.glob(Path + '/*.png'))]
plt.imshow(images[1])
im=[]
for i in Path:
...  for j in sorted(glob.glob(Path + '/*.png')):
...   im.append(j)
... 
>>> r=[]
>>> for k in im:
...  r.append(cv2.imread(k, cv2.IMREAD_GRAYSCALE))
... 
>>> plt.imshow(r[1])
<matplotlib.image.AxesImage object at 0x7f9baae08050>
>>> plt.show()
>>> rim=[]
>>> for i in r:
...  res=cv2.resize(i,(32,32),interpolation=cv2.INTER_AREA)
...  rim.append(res)
'''

group_chars = list(zip(*[iter(charsr)] * 7))
group_col = list(zip(*[iter(r0)] * 33))
group_label=[]
for i in text:
    if i==" " or i=='\n':continue
    for j in range(len(group_chars)):
        if i in(group_chars[j]): group_label.append(j)
group_label_col=[]
for i in text:
    if i==" " or i=='\n' or i=='':continue
    for j in range(len(group_col)):
        if y[i] in(group_col[j]): group_label_col.append(j)
#change the lable to array
#label=numpy.array([numpy.array(tex).flatten() for tex in charlable])
labelr=group_label

labelc=group_label_col

#convert the text as array
#label=numpy.array([numpy.array(open(tex.decode('utf-8')),'f').flatten() for tex in textPath])
#convert image to gray resize and then flaten the size of the image in to 1*row or by columan
im_array = np.array( [np.array(Image.open(img).resize((img_rows,img_cols)), 'f').flatten() for img in imagePath])
print len(im_array)
data, labelsr,labelsc =shuffle (im_array, labelr,labelc)

#datar, labelsr = shuffle(im_array, labelgr, random_state=2)
#data, labels = shuffle(im_array, labelrc, random_state=2)
training=[data,labelsr,labelsc]
(x_train,y_trainr,y_trainc)=(training[0],training[1],training[2])

# batch_size to train
batch_size =256
# number of output classes
nb_clas_r = 33
nb_clas_c=7
# number of epochs to train
nb_epoch =15

#print label[2515:2520]
#print label[395:465]
# STEP 1: split X and y into training and testing sets

X_train, X_test, Y_trainr, Y_testr,Y_trainc, Y_testc = train_test_split(x_train, y_trainr, y_trainc, test_size=0.1, random_state=1)

#words, we want to transform our dataset from having shape (n, width, height) to (n, depth, width, height)..this is for theano backend
#but for tensorflow backedn we will have(n,width,height,depth)
#if bk.image_data_format() == 'channels_first':
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
#Y_train =np_utils.to_categorical(y_train, nb_clas_r)
#Y_test = np_utils.to_categorical(y_test, nb_classgr)

Y_trainr =np_utils.to_categorical(Y_trainr, nb_clas_r)
Y_testr = np_utils.to_categorical(Y_testr, nb_clas_r)

Y_trainc =np_utils.to_categorical(Y_trainc, nb_clas_c)
Y_testc = np_utils.to_categorical(Y_testc, nb_clas_c)

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
out_r =Dense(nb_clas_r ,activation="softmax")(inner)
out_c =Dense(nb_clas_c,activation="softmax")(inner)

model=Model(inputs=input_data, outputs=[out_r, out_c])

model.summary()
#++++++++++++++++++++++++++++++++++++++++loss
 from keras.preprocessing.image import ImageDataGenerator
 datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
     rotation_range=20,
     width_shift_range=0.2,
     height_shift_range=0.2,
     horizontal_flip=True)

model.compile(loss=['categorical_crossentropy','categorical_crossentropy'], optimizer='ADAM', metrics=['accuracy'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
#model.fit(x, y, validation_split=0.2, callbacks=[early_stopping])
hist = model.fit(X_train,[Y_trainr,Y_trainc], batch_size=batch_size, epochs=15,verbose=1, validation_split=0.2,shuffle=False, callbacks=[early_stopping])

'''# to save and load models that we creat ones
from keras.models import  load_model
model.save_weights("model.h5")

hist = model.fit(datagen.fit(X_train,[Y_trainr,Y_trainc]), batch_size=batch_size,steps_per_epoch=X_train // batch_size, epochs=15,verbose=1, validation_split=0.2,shuffle=False, callbacks=[early_stopping])

print("Saved model to disk")
# load weights into new model
k=load_model("model.h5")
print("Loaded model from disk")'''
#testing model so as to predict the class using testset with out class.
classs=model_group.predict_classes(X_test)# dispaly the  predicted class
#print  y_test
#print sorted(classs)
#print classs
score = model.evaluate(X_test,[Y_testr,Y_testc],batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

x=model_row_col.pred(x_test)
clas=x.argmax(axis=-1)
''''
pred = model.predict(X_test)[0]
y_pred = [np.argmax(p) for p in pred]
y_true = [np.argmax(p) for p in Y_test]
from sklearn.metrics import accuracy_score
accuracy_score(y_test,classs)'''
row_acc=0.9450624957680702
col_acc=0.9418749988079071
score[3]
Out[139]: 0.9522499904036522 ---row
score[4]
Out[140]: 0.9498749941587448--col
Out[150]: 0.930375--overall

score[3]
Out[154]: 0.9522499948740005
score[4]
Out[155]: 0.9507499977946281
Out[163]: 0.930625
score[3]
Out[210]: 0.9555499890327454
score[4]
Out[211]: 0.9548749903440475
inter
Out[221]: 0.93924

Out[44]: 0.9661538457259153
score[4]
Out[45]: 0.9596153843097198
inter
Out[53]: 0.9497179487179487


pred = model.predict(X_test)[0]
y_predr = [np.argmax(p) for p in pred]
y_true = [np.argmax(p) for p in Y_testr]
yr=[]
for i in range(len(Y_testr)):
 if y_predr[i]==y_true[i]: yr.append(i)


pred = model.predict(X_test)[1]
y_predc = [np.argmax(p) for p in pred]
y_truec = [np.argmax(p) for p in Y_testc]
yc=[]
for i in range(len(Y_testc)):
 if y_predc[i]==y_truec[i]: yc.append(i)
inter=float(len(set(yr).intersection(yc)))/len(Y_testr)
 inter=
Out[99]: 0.9279375




['loss',
 'val_dense_4_loss',
 'val_dense_3_acc',
 'dense_4_loss',
 'dense_4_acc',
 'dense_3_acc',
 'dense_3_loss',
 'val_dense_4_acc',
 'val_dense_3_loss',
 'val_loss']

hist.history.values()

hist.history.keys()

trr_acc = hist.history['dense_3_acc']
valr_acc = hist.history['val_dense_3_acc']
trc_acc = hist.history['dense_4_acc']
valc_acc = hist.history['val_dense_4_acc']



# to print training  accuracy vs validation accuracy
plt.plot(epochs, trr_acc)
plt.plot(epochs, valr_acc)
plt.plot(epochs, trc_acc)
plt.plot(epochs, valc_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train-row-predictor', 'val-row-predictor','train-column-predictor', 'val-column-predictor'], loc=4)
plt.show()


# visualizing  accuracy per epoch
train_acc = histr.history['acc']
val_acc = histcr.history['val_acc']
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
def joint(X, y1, y2):
    genX1 = gen.flow(X,y1,  batch_size=batch_size,seed=666)
    genX2 = gen.flow(X,y2, batch_size=batch_size,seed=666)
    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            #Assert arrays are equal - this was for peace of mind, but slows down training
            #np.testing.assert_array_equal(X1i[0],X2i[0])
            yield [X1i[0], X2i[1]], X1i[1]

gen_flow = joint(X_train, Y_trainr, Y_trainc)
plot_model(model,show_shapes=True, to_file='/home/belay/PycharmProjects/untitled/ocr/amahric_model.png')


#real data(overall accuracy )
y_true = np.loadtxt("/home/nbm/aa_class/y_test", delimiter="\n")
y_pred_g = np.loadtxt("/home/nbm/aa_class/clas_g", delimiter="\n")
y_pred_row = np.loadtxt("/home/nbm/aa_class/clas_row", delimiter="\n")
pred_row=[]
for i in range(len(y_true)):
 if y_pred_row[i]==y_true[i]: pred_row.append(i)
y=zip(*(iter(range(231)),) * 7)
t=[] # to convert the row col in to group level .
for i in range(len(y_true)):
 for j in range(len(y)):
  if y_true[i] in y[j]:t.append(j)
pred_gr=[]
for i in range(len(t)):
 if t[i]==y_pred_g[i]:pred_gr.append(i)
inter=float(len(set(pred_gr).intersection(pred_row)))/len(y_true)

