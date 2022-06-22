
import os
#os.environ['KERAS_BACKEND'] = 'theano'
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
from keras.layers import Input, Dense,Bidirectional, Activation,BatchNormalization,Permute,Flatten,Masking,GaussianNoise,Reshape, Lambda,TimeDistributed,Dropout
from keras.layers import TimeDistributed, Activation, Dense, Input, Bidirectional, LSTM, Masking, GaussianNoise
from keras.layers.merge import add, concatenate
from keras.models import Model,load_model
from keras.layers.recurrent import GRU,LSTM
from keras.optimizers import SGD, adam
from PIL import Image
from keras.preprocessing import sequence
import  glob,cv2
import numpy as np
import editdistance
from sklearn.model_selection import train_test_split

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
#======================================================================================
img_row = 128
img_col = 48
num_class=281# number of charcter in the total text+1

# inner = ctc_convolutional_func(input_data)
# conv_to_rnn_dims = inner.get_shape()
'''
path1='/home/belay/PycharmProjects/ocropy-master/tests/output/new_2000/powergeez'
imagepath1 = sorted(glob.glob(path1 + '/*.bin.png'))
textpath1 = sorted(glob.glob(path1 + '/*.gt.txt'))
path2='/home/belay/PycharmProjects/ocropy-master/tests/output/new_synth_last/new_synth_last'
imagepath2 = sorted(glob.glob(path2 + '/*.bin.png'))
textpath2 = sorted(glob.glob(path2 + '/*.gt.txt'))
path3='/home/belay/PycharmProjects/ocropy-master/tests/output/synth_100/synth_100'
imagepath3 = sorted(glob.glob(path3 + '/*.bin.png'))
textpath3 = sorted(glob.glob(path3 + '/*.gt.txt'))
'''
path1 = '/home/belay/PycharmProjects/ocropy-master/tests/output/sentence_label/sentence'  # path of folder of images
# path2 = /home/belay/PycharmProjects/2nd_round/datasets/word_label'
imagepath = sorted(glob.glob(path1 + '/*.bin.png'))
textpath = sorted(glob.glob(path1 + '/*.gt.txt'))
out = open("/home/belay/PycharmProjects/2nd_round/icdar_dataset/tint.txt", "w")
for name in textpath:
     with open(name) as f:
         for line in f:
             if line == " " or line == 'u' or len(line)==0 : continue
             # print(line.decode('utf-8').split())
             out.write(line)
 out.close()
#text2 = open('/home/belay/PycharmProjects/2nd_round/icdar_dataset/powergeez.txt', 'r').read().decode('utf-8')
#text2 = open('/home/belay/PycharmProjects/2nd_round/icdar_dataset/new_synth_last.txt', 'r').read().decode('utf-8')
text = open('/home/belay/PycharmProjects/ocropy-master/tests/output/sentence_label/sentence_label.txt', 'r').read().decode('utf-8')  # strip use to remove the last \n line
lines = text.split('\n')
chars = sorted(list(set(text)))
mapping = dict((c, i) for i, c in enumerate(chars))  # map charcter to index
mapping1 = dict((i, c) for i, c in enumerate(chars))  # mapp index to character

sequences = list()
for line in lines:
    # integer encode line
    if len(line)==0: continue
    encoded_seq = [mapping[char] for char in line]
    # store
    sequences.append(encoded_seq)
label = [list(line) for line in sequences]
# to change each word in to independent array list that have its own size which depends on the character it contains.
#maxlen=41
maxlen=32# since the max string length in printed data which have 32 charcters
maxlen = max((len(r)) for r in sequences)  # find max length(longest  line in the array of the array in the list
# addopted from: https://machinelearningmastery.com/data-preparation-variable-length-input-sequences-sequence-prediction/
label = np.asarray([np.pad(r, (0, maxlen - len(r)), 'constant', constant_values=0) for r in label])
# pad -1 around the shortest text line
# label=label.reshape(label[0],maxlen)
#==========================================
text = open('/home/belay/PycharmProjects/2nd_round/datasets/w.txt', 'r').read().decode('utf-8').split('\n')
x= len(set(text)) 

#========== if it is from numpy array+++++++================
y_train=np.load('/home/belay/PycharmProjects/2nd_round/icdar_dataset/all_dataset/y_tran.npy')
u= len(np.unique(x, axis=0))



def im_resize(myimage):
  (wt,ht)=(img_row,img_col)
  (h,w)=myimage.shape
  fw=float(w)/wt
  fh=float(h)/ht
  f=max(fw,fh)
  newsize = (max(min(wt, int(w / f)), 1), max(ht, 1))
  myimage = cv2.resize(myimage, newsize)
  target = np.ones([ht, wt]) * 255
  target[0:newsize[1], 0:newsize[0]] = myimage
  newimage =cv2.transpose(target)
  (m, s) = cv2.meanStdDev(newimage)
  m=[0][0]
  s=s[0][0]
  newimage=newimage-m
  newimage=newimage/s if s>0 else newimage
  return newimage

#
# image = np.asarray([np.array(Image.open(img).resize((img_row, img_col)), 'f').flatten() for img in imagepath])
# image = image.reshape(image.shape[0], img_row, img_col, 1)
# image = image.astype('float32')
# label= label.reshape(len(image),maxlen)
# label=label.astype('float32')
#padding_value=255
image = [np.array(Image.open(i), 'f') for i in imagepath]#(256,48)
#image2 = [np.array(Image.open(i), 'f') for i in imagepath[99999:]
img = []
for i in image:
    #imgr=cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)# it is not necessay sinmce it is gray scale image synthericaly generated
    #im1=cv2.resize(i,(img_col,img_row),interpolation=cv2.INTER_AREA)
    #imgt=im1img/255
     img.append(im_resize(i))
    #img.append(np.transpose(im_resize(i),(1,0)))# to shange the sape of the image in to bs,wedth,hegith
#for i in image2:
    #imgr=cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)# it is not necessay sinmce it is gray scale image synthericaly generated
    #im1=cv2.resize(i,(img_col,img_row),interpolation=cv2.INTER_AREA)
    #imgt=im1img/255
    #img.append(np.transpose(im_resize(i),(1,0)))  
    #img[1].shape  [500,48]
#im = sequence.pad_sequences(img, value=float(255), dtype='float32',
                            #padding="post", truncating='post')
im=np.array(img)
#im=np.transpose(im,(0,2,1))
X_train, X_test, y_train, y_test = train_test_split(im, label, test_size=0.05)
#X_train.shape
x=np.load('X_train.npy')
y=np.load('y_train.npy')
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.05)
# ========================================================

nub_features = len(X_train[0][0])

nb_train = len(X_train)
nb_val = len(X_val)
#nb_test = len(X_test)
x_train_len = np.asarray([len(X_train[i]) for i in range(nb_train)])
y_train_len = np.asarray([len(y_train[i]) for i in range(nb_train)])
x_val_len = np.asarray([len(X_val[i]) for i in range(nb_val)])
y_val_len = np.asarray([len(y_val[i]) for i in range(nb_val)])


y_train_len = np.asarray([len(y_train[i]) for i in range(nb_train)])
y_test_len = np.asarray([len(y_test[i]) for i in range(nb_test)])
# ========================================================
#t = int(len(X_train) * 0.95)
#def ctc_convolutional_func():
rnn_size = 128
act= 'relu'
#input_shape = (252,48,1)
#input_data = Input(name='the_input', shape=input_shape, dtype='float32')
input_data = Input(name='input', shape=(None, nub_features)) # nb_features = image height

#masking = Masking(mask_value=padding_value)(input_data)
#noise = GaussianNoise(0.01)(masking)
#inner = Reshape(target_shape=(int(conv_to_rnn_dims[1]), int(conv_to_rnn_dims[2] * conv_to_rnn_dims[3])))(inner)#change in to two dimensional from three dimensional

    # cuts down input size going into RNN:
#inner = Dense(time_dense_size, activation='relu', name='dense1')(inner)
rnn_1 = LSTM(rnn_size, kernel_initializer="he_normal", return_sequences=True)(input_data)
rnn_1b = LSTM(rnn_size, kernel_initializer="he_normal", go_backwards=True, return_sequences=True)(input_data)
rnn1_merged = add([rnn_1, rnn_1b])#It takes as input a list of tensors, all of the same shape, and returns a single tensor (also of the same shape).

rnn_2 = LSTM(rnn_size, kernel_initializer="he_normal", return_sequences=True)(rnn1_merged)
rnn_2b = LSTM(rnn_size, kernel_initializer="he_normal", go_backwards=True, return_sequences=True)(rnn1_merged)
inner = Dense(num_class, kernel_initializer='he_normal',
              name='dense2')(concatenate([rnn_2, rnn_2b]))
out_clas = Activation('softmax', name='softmax')(inner)


labels = Input(name='the_labels', shape=[32], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')
#predlabel= ctc_lstm_func()
# ================================================================================================================================
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc_loss')([out_clas, labels, input_length, label_length])

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
pmodel = Model(inputs=input_data, outputs=out_clas)
tmodel = Model(inputs=[input_data, labels, input_length, label_length],outputs=loss_out)
# Model is the keras  liberary model and input_data is the input laye of the nural network
tmodel.compile(loss={'ctc_loss': lambda y_true, y_pred: y_pred}, optimizer=adam(lr=0.001))
# =======================================================================================================================
#out_clas=ctc_lstm_func()
# decode_out= Lambda(ctc_decoding_lambda_func, output_shape=(20,1), name='ctc_decode')([out_clas,input_length])
tmodel.get_weights()
pmodel.get_weights()
# predmodel=Model(inputs=[input_data,input_length],outputs=decode_out)
#model.compile(loss={'ctc_decode':lambda  y_true,y_pred:y_pred},optimizer=sgd)
test_func = K.function([input_data], [out_clas])


hist=tmodel.fit([X_train, y_train, x_train_len,y_train_len], np.zeros(len(y_train)),
                batch_size=200,
                epochs=50,
                verbose=1,
                #shuffle=True,
                validation_data=([X_val, y_val,x_val_len,y_val_len], np.zeros(len(y_val))))
#return model
#pmodel.save_weights('pmodelsynthsent_100.h5')
pmodel.save('modelsynthword.h5')
model = load_model('pmodelsynthsent_100.h5')
#++++++++++++++++++++++++++++++++++++++=
#not: when we train the model tmodel both pmodel and tmodel wieths are updated.(same weigth for both model(i.e share their weigth)
plt.plot(epochs, train_loss, 'b--')
plt.plot(epochs, val_loss, 'r--')
plt.xlabel('Num of Epochs')
plt.ylabel('Loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train_loss', 'val_loss'], loc='right center')
plt.show()
'''
once we load image from differtn folder then we change the in to numpy array a
the we conctnet as floowes both the input data and the ground truth along axis 0.
it can be done after train_test split done for momory management.

imp=np.load('/home/belay/PycharmProjects/2nd_round/icdar_dataset/all_dataset/X_trainp.npy')
impg=np.load('/home/belay/PycharmProjects/2nd_round/icdar_dataset/all_dataset/X_train_pg.npy')
im_100=np.load('/home/belay/PycharmProjects/2nd_round/icdar_dataset/all_dataset/X_train_100.npy')

mapping=open('/home/belay/PycharmProjects/2nd_round/icdar_dataset/all_dataset/test/mapping.txt','r')
mapping=eval(mapping.read())
imt=np.append(im1,im2,axis=0)

y_pred=pmodel.predict(X_test)
predmodel=ctc_convolutional_func()
y_pred=predmodel.predict(X_test)
print y_pred

train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.plot(epochs, train_loss, 'b--')
plt.plot(epochs, val_loss, 'r--')
plt.legend(['train_loss', 'val_loss'], loc='upper right')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.show()

'''

out = K.get_value(K.ctc_decode(y_pred[:, :, :], input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1])[0][0])
#[:,: maxlen]


for i in range(10):
    # print the 10 first predictions
    print("Prediction :", [j for j in out[i] if j != -1], " -- Label : ", y_test[i])

for i in range(10):
    # print the 10 first predictions
    print("Prediction :", [j for j in out[i] if j not in (0,-1)], " -- Label : ", [k for k in y_testpg[i] if k!=0])

for i in range(len(chars)):
    print (str(i)+ " "+ chars[i].encode('utf-8')+"\n")
for i in range(10):
    # print the 10 first predictions
    print("Prediction :", [j for j in out[i] if j not in (0,-1)])
    print("Prediction :", [k for k in y_testpg[i] if k!=0])

#===========edit distance exampl====================================
import  editdistance
true=['abebe','belay','almaze']
pred=['abebee','belau','almaaze']
true=[]# to stor value of charcter by removing zeero which was padded prvieously and also this is the value of newline in the test label
for i in range(len(y_test)):
    x=[j for j in y_test[i] if j!=0]
    true.append(x)
pred=[]# to store the predicted charcter except zerro and -1 which are padded value nad blach soace predicted during testing
for i in range(len(out)):
    x=[j for j in out[i] if j not in(0,-1)]
    pred.append(x)
cer=0
for(i,j) in zip(true,pred):
    x=editdistance.eval(i,j)
    cer=cer+x
err=cer
x=0
for i in range(len(true)):
    x=x+len(true[i])
totalchar=x
cerp=(float(err)/totalchar)*100 # to calculate character error rate in percent
''''
y=[]
k=0
for(i,j) in zip(true,pred):
    x=editdistance.eval(i,j)
    if x<5: 
        y.append(k) 
        #print(k,x)
    k=k+1 
newx=[]
for i in y:
    newx.append(x_test[i])

x=np.array(newx)
   
imt=np.append(im1,im2,axis=0)# to concatnet two numpy array(in the same way we can concatnet as much as we we want

x=np.load('X_train.npy')
y=np.load('y_train.npy')
X_train, X_val, y_train, y_val = train_test_split(x,y, test_size=0.05)
X_val=X_train[t:]
y_val=y_train[t:]
X_trainv=X_train[:t]
y_trainv=y_train[:t]

np.save('/home/belay/PycharmProjects/2nd_round/icdar_dataset/all_dataset/X_trainp',X_trainp)
np.save('/home/belay/PycharmProjects/2nd_round/icdar_dataset/all_dataset/y_trainp',y_trainp)
np.save('/home/belay/PycharmProjects/2nd_round/icdar_dataset/all_dataset/X_testp',X_testp)
np.save('/home/belay/PycharmProjects/2nd_round/icdar_dataset/all_dataset/y_testp',y_testp)

np.save('X_train',np.array(X_train))
np.save('y_train',y_train)
np.save('X_test',X_test)
np.save('y_test',y_test)
np.save('X_val',X_val)
np.save('y_val',np.array(y_val))
# to load e.g
x=np.load('X_train.npy')
X_test=np.load('/home/belay/PycharmProjects/2nd_round/X_test.npy')#to retrive from differnt path

plt.imshow(x[1]) # to diplay the  sample image with yellow background since it used colormap to map intensity to color.
plt.imshow(x[1],cmap='Greys_r')# todisply the backgrouund to be white
'''


def decode_predict_ctc(out, top_paths=1):
    results = []
    beam_width = 7
    if beam_width < top_paths:
        beam_width = top_paths
    for i in range(top_paths):
        lables = K.get_value(K.ctc_decode(out, input_length=np.ones(out.shape[0]) * out.shape[1],greedy=True, beam_width=beam_width, top_paths=top_paths)[0][i])[0]
        #text = labels_to_text(chars, lables)
        results.append(lables)
    return results

def decode(labels):
    # print (labels)
    ret = []
    # print (type(labels))
    lab = [int(x) for x in labels]
    for c in lab:
        if c == len(alphabet):  # CTC Blank
            ret.append("")
        else:
            ret.append(alphabet[c])
    return "".join(ret)


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def decode_batch(test_func, word_batch):
    out = test_func([word_batch])[0]
    ret = []
    for j in range(out.shape[0]):
        out_best = list(
            np.argmax(out[j, 2:], 1))  # return indexes of highest probability letters at each point/timestep
        out_best = [k for k, g in itertools.groupby(out_best)]  # removes adjacent duplicate indexes
        # print (out_best)
        outstr = decode(out_best)
        ret.append(outstr)
    return ret
input_data = Input(name='input', shape=(None, nub_features)) 

noise = GaussianNoise(0.01)(input_data)
blstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1))(noise)
blstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1))(blstm)
blstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1))(blstm)
dense = TimeDistributed(Dense(num_class, name="dense"))(blstm)
out_clas = Activation('softmax', name='softmax')(dense)
hist=tmodel.fit([X_train, y_train, x_train_len,y_train_len], np.zeros(len(y_train)),
                batch_size=300,
                epochs=20,
                verbose=1,
                #shuffle=True,
                validation_data=([X_val, y_val,x_val_len,y_val_len], np.zeros(len(y_val))))




