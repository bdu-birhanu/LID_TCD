import os
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation,BatchNormalization,Permute,Flatten,Masking,GaussianNoise,Reshape, Lambda,TimeDistributed,Dropout,Bidirectional,ZeroPadding2D
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import GRU,LSTM
from keras.optimizers import SGD,Adam
from PIL import Image
from keras.preprocessing import sequence
import  glob,cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
img_row = 256
img_col = 32
num_class=281

def im_resize(myimage):
  (wt,ht)=(img_row,img_col)#image row and col
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
#======================================================================================


# inner = ctc_convolutional_func(input_data)
# conv_to_rnn_dims = inner.get_shape()
path1 = '/home/belay/PycharmProjects/2nd_round/datasets/printed_im_tex/im'  # path of folder of images
# path2 = '/home/belay/english_word'
imagepath = sorted(glob.glob(path1 + '/*.bin.png'))
textpath = sorted(glob.glob(path1 + '/*.gt.txt'))
out = open("/home/belay/PycharmProjects/2nd_round/datasets/printed_im_tex/textr.txt", "w")
for name in textpath:
    with open(name) as f:
        for line in f:
            if line == " " or line == 'u': continue
             # print(line.decode('utf-8').split())
            out.write(line)
out.close()

text = open('/home/belay/PycharmProjects/2nd_round/datasets/printed_im_tex/textr.txt', 'r').read().decode('utf-8')  # strip use to remove the last \n line
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

maxlen = max((len(r)) for r in sequences)  # find max length(longest  line in the array of the array in the list
# addopted from: https://machinelearningmastery.com/data-preparation-variable-length-input-sequences-sequence-prediction/
label = np.asarray([np.pad(r, (0, maxlen - len(r)), 'constant', constant_values=0) for r in label])
# pad -1 around the shortest text line
# label=label.reshape(label[0],maxlen)

#
# image = np.asarray([np.array(Image.open(img).resize((img_row, img_col)), 'f').flatten() for img in imagepath])
# image = image.reshape(image.shape[0], img_row, img_col, 1)
# image = image.astype('float32')
# label= label.reshape(len(image),maxlen)
# label=label.astype('float32')
#padding_value=255
image = [np.array(Image.open(i), 'f') for i in imagepath]
img = []
for i in image:
    imgr=cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)# it is not necessay since it is gray scale image synthericaly generated
    im1=im_resize(imgr)
    #imt=im1.T
    #im2=np.expand_dims(im1,axis=3)
    img.append(im1)
    #img[1].shape  [500,48]
#im = sequence.pad_sequences(img, value=float(255), dtype='float32',
                            #padding="post", truncating='post')
im=np.array(img)# (samplesize,w,h,channel)
X_train, X_test, y_train, y_test = train_test_split(im, label, test_size=0.1)
#nub_features = len(X_train[0][0])
# ========================================================
nb_train = len(X_train)
nb_test = len(X_test)

#nb_features = len(X_train[0][0])
# create list of input lengths

# x_train_len = np.asarray([len(X_train[i]) for i in range(nb_train)])
# x_test_len = np.asarray([len(X_test[i]) for i in range(nb_test)])
# y_train_len = np.asarray([len(y_train[i]) for i in range(nb_train)])
# y_test_len = np.asarray([len(y_test[i]) for i in range(nb_test)])

t = int(len(X_train) * 0.95)
#def ctc_convolutional_func():
rnn_size = 256
act= 'relu'
#input_shape = (252,48,1)
#input_data = Input(name='the_input', shape=input_shape, dtype='float32')
#input_data = Input(name='input', shape=(None, nub_features)) # nb_features = image height

# masking = Masking(mask_value=padding_value)(input_data)
# noise = GaussianNoise(0.01)(masking)
#inner = Reshape(target_shape=(int(conv_to_rnn_dims[1]), int(conv_to_rnn_dims[2] * conv_to_rnn_dims[3])))(inner)#change in to two dimensional from three dimensional

    # cuts down input size going into RNN:
#inner = Dense(time_dense_size, activation='relu', name='dense1')(inner)

input = Input(shape=(img_row,img_col,1), dtype='float32')#width and hiegth  respectivly
m = Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same', name='conv1')(input)
m = MaxPooling2D(pool_size=(2, 2), name='pool1')(m)
m = Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same', name='conv2')(m)
# m = MaxPooling2D(pool_size=(2, 2), name='pool2')(m)
#conv_to_rnn_dims = (img_w // (self.pool_size ** 2), (self.img_h // (self.pool_size ** 2)) * self.conv_filters)
conv=m.get_shape()
conv=(int(conv[1]),int(conv[2]*int(conv[3])))
inner = Reshape(target_shape=conv, name='reshape')(m)

# cuts down input size going into RNN:
inner = Dense(32, activation=act, name='dense1')(inner)

# m = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='conv3')(m)
# m = Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', name='conv4')(m)
# m = ZeroPadding2D(padding=(0, 1))(m)
# m = MaxPooling2D(pool_size=(2, 2), strides=(2, 1), padding='valid', name='pool3')(m)
#
# m = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='conv5')(m)
# m = BatchNormalization(axis=1)(m)
# m = Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same', name='conv6')(m)
# m = BatchNormalization(axis=1)(m)
# m = ZeroPadding2D(padding=(0, 1))(m)
# m = MaxPooling2D(pool_size=(2, 2), strides=(2, 1), padding='valid', name='pool4')(m)
# m = Conv2D(512, kernel_size=(2, 2), activation='relu', padding='valid', name='conv7')(m)
#
# m = Permute((2, 1, 3), name='permute')(m) # shape becoms in width and hiegth respctvily
# m = TimeDistributed(Flatten(), name='timedistrib')(m)# flatten menain the image ic converted in to h*w and number of filter

rnn_1 = LSTM(rnn_size, kernel_initializer="he_normal", return_sequences=True)(inner)
rnn_1b = LSTM(rnn_size, kernel_initializer="he_normal", go_backwards=True, return_sequences=True)(inner)
rnn1_merged = add([rnn_1, rnn_1b])#It takes as input a list of tensors, all of the same shape, and returns a single tensor (also of the same shape).

rnn_2 = LSTM(rnn_size, kernel_initializer="he_normal", return_sequences=True)(rnn1_merged)
rnn_2b = LSTM(rnn_size, kernel_initializer="he_normal", go_backwards=True, return_sequences=True)(rnn1_merged)
fc = Dense(num_class, kernel_initializer='he_normal',
              name='dense2')(concatenate([rnn_2, rnn_2b]))
out_clas = Activation('softmax', name='softmax')(fc)
# m = Bidirectional(GRU(rnn_size, return_sequences=True), name='blstm1')(m)
# m = Dense(rnn_size, name='blstm1_out', activation='linear')(m)
# m = Bidirectional(GRU(rnn_size, return_sequences=True), name='blstm2')(m)
# out_clas = Dense(num_class, name='blstm2_out', activation='softmax')(m)

labels = Input(name='the_labels', shape=[41], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')
#predlabel= ctc_lstm_func()
# ================================================================================================================================
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc_loss')([out_clas, labels, input_length, label_length])

#sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=1e-6)
pmodel = Model(inputs=input, outputs=out_clas)
tmodel = Model(inputs=[input, labels, input_length, label_length],outputs=[loss_out])
# Model is the keras  liberary model and input_data is the input laye of the nural network
tmodel.compile(loss={'ctc_loss': lambda y_true, y_pred: y_pred}, optimizer=adam)
# =======================================================================================================================
#out_clas=ctc_lstm_func()
# decode_out= Lambda(ctc_decoding_lambda_func, output_shape=(20,1), name='ctc_decode')([out_clas,input_length])
tmodel.get_weights()
pmodel.get_weights()
# predmodel=Model(inputs=[input_data,input_length],outputs=decode_out)
#model.compile(loss={'ctc_decode':lambda  y_true,y_pred:y_pred},optimizer=sgd)
test_func = K.function([input_data], [out_clas])
t=31100# is around 10% of the validation  and 90% of traininjg

x_train_len=np.ones(len(X_train))*int(conv[1]-2)
y_train_len=np.ones(len(X_train))*41
def next_batch(self):
        while True:
            # width and height are backwards from typical Keras convention
            # because width is the time dimension when it gets fed into the RNN
            if K.image_data_format() == 'channels_first':
                X_data = np.ones([self.batch_size, 1, self.img_w, self.img_h])
            else:
                X_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])
            Y_data = np.ones([self.batch_size, self.max_text_len])
            input_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor - 2)
            label_length = np.zeros((self.batch_size, 1))
            source_str = []
                                   
            for i in range(self.batch_size):
                img, text = self.next_sample()
                img = img.T
                if K.image_data_format() == 'channels_first':
                    img = np.expand_dims(img, 0)
                else:
                    img = np.expand_dims(img, -1)
                X_data[i] = img
                Y_data[i] = text_to_labels(text)
                source_str.append(text)
                label_length[i] = len(text)
                
            inputs = {
                'the_input': X_data,
                'the_labels': Y_data,
                'input_length': input_length,
                'label_length': label_length,
                #'source_str': source_str
            }
            outputs = {'ctc': np.zeros([self.batch_size])}
            yield (inputs, outputs)
hist=tmodel.fit([X_train[:t], y_train[:t], x_train_len[:t],y_train_len[:t]], np.zeros(len(y_train))[:t],
                batch_size=100,
                epochs=10,
                verbose=1,
                shuffle=True,
                validation_data=([X_train[t:], y_train[t:],x_train_len[t:],y_train_len[t:]], np.zeros(len(y_train))[t:]))
#return model
pmodel.save_weights('modelprintedsentence.h5')
#++++++++++++++++++++++++++++++++++++++=
#not: when we train the model tmodel both pmodel and tmodel wieths are updated.(same weigth for both model(i.e share their weigth)
pmodel.predict(X_test)
predmodel=ctc_convolutional_func()
y_pred=predmodel.predict(X_test)
print y_pred

out = K.get_value(K.ctc_decode(y_pred[:, :, :], input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1])[0][0])[:,: maxlen]


for i in range(10):
    # print the 10 first predictions
    print("Prediction :", [j for j in out[i] if j != -1], " -- Label : ", y_test[i])


#===========edit distance exampl====================================
import  editdistance
true=['abebe','belay','almaze']
pred=['abebee','belau','almaaze']
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

