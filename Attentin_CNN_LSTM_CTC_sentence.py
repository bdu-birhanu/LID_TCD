import os
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation,BatchNormalization,Permute,Flatten,Masking,GaussianNoise,Reshape, Lambda,TimeDistributed,Dropout
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import  Merge, RepeatVector, Dot, Bidirectional,LSTM,Input, Dense, Activation,Lambda,Concatenate,Conv2D,MaxPool2D
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, MaxPool2D
from keras.models import load_model
import  cv2
import numpy as np
import  editdistance
from sklearn.model_selection import train_test_split


num_class=280# number of charcter in the total text

train_x_i=np.load('/home/belay/PycharmProjects/3rd_round/text_line/train128by32/X_trainp_pg_vg.npy')# [sample size,32,128]
train_y_t=np.load('/home/belay/PycharmProjects/3rd_round/text_line/train128by32/y_trainp_pg_vg.npy')
test_x_pg = np.load('/home/belay/PycharmProjects/3rd_round/text_line/test128by32/X_test_pg.npy')
test_y_pg=np.load('/home/belay/PycharmProjects/3rd_round/text_line/test128by32/y_test_pg.npy')
test_xp  = np.load('/home/belay/PycharmProjects/3rd_round/text_line/test128by32/X_testp.npy')
test_y_p=np.load('/home/belay/PycharmProjects/3rd_round/text_line/test128by32/y_testp.npy')
test_x_vg = np.load('/home/belay/PycharmProjects/3rd_round/text_line/test128by32/X_test_vg.npy')
test_y_vg=np.load('/home/belay/PycharmProjects/3rd_round/text_line/test128by32/y_test_vg.npy')
#==============================================================================================================


#val_tex=np.load('/home/belay/PycharmProjects/3rd_round/word_synth/y_val.npy')
train_x,val_x, train_y, val_y = train_test_split(train_x_i, train_y_t, test_size=0.07)
#reshape the size of the image from 3D to 4D so as to make the input size is similar with it.
X_train=train_x.reshape(train_x.shape[0],train_x.shape[1],train_x.shape[2],1) #[samplesize,32,128,1]
X_test_pg=test_x_pg.reshape(test_x_pg.shape[0],test_x_pg.shape[1],test_x_pg.shape[2],1)
X_test_vg=test_x_vg.reshape(test_x_vg.shape[0],test_x_vg.shape[1],test_x_vg.shape[2],1)
X_test_p=test_xp .reshape(test_xp .shape[0],test_xp .shape[1],test_xp.shape[2],1)
X_val=val_x.reshape(val_x.shape[0],val_x.shape[1],val_x.shape[2],1)
y_train=train_y
# y_testp=test_texp
y_val=val_y


# ========================================================
nb_train = len(X_train)
# nb_testp = len(X_testp)
nb_val = len(X_val)

#nb_features = len(X_train[0][0])
# create list of input lengths
x_train_len = np.array([len(X_train[i])+31 for i in range(nb_train)]) #the +31 here is just to make the size of the image equal to the input of LSTM
# x_test_len = np.asarray([len(X_testp[i]) for i in range(nb_testp)])#the +31 here is just to make the size of the image equal to the out put of LSTM
x_val_len = np.array([len(X_val[i])+31 for i in range(nb_val)])#the +31 here is just to make the size of the image equal to the out put of LSTM
y_train_len = np.array([len(y_train[i]) for i in range(nb_train)])
# y_test_len = np.asarray([len(y_testp[i]) for i in range(nb_testp)])
y_val_len = np.array([len(y_val[i]) for i in range(nb_val)])

dot_pro = Dot(axes = 1)
def attention(lstm_out):
    # hidden_states.shape = (batch_size, time_steps, hidden_size)
    #hidden_size = int(lstm_out.shape[2])
    # _t stands for transpose
    hidden_state_t = lstm_out
    # hidden_states_t.shape = (batch_size, hidden_size, time_steps)
    # this line is not useful. It's just to know which dimension is what.
     #hidden_state_t = Reshape((hidden_size, 31), name='attention_input_reshape')(hidden_states_t)
    # Inside dense layer
    # a (batch_size, hidden_size, time_steps) dot W (time_steps, time_steps) => (batch_size, hidden_size, time_steps)
    # W is the trainable weight matrix of attention
    # Luong's multiplicative style score
    score = Dense(63, activation='tanh', name='attention_score_vec')(hidden_state_t)
    #score_first_part_t = Permute((2, 1), name='attention_score_vec_t')(score_first_part)
    #            score_first_part_t         dot        last_hidden_state     => attention_weights
    # (batch_size, time_steps, hidden_size) dot (batch_size, hidden_size, 1) => (batch_size, time_steps, 1)
    # h_t = Lambda(lambda x: x[:, :, -1], output_shape=(hidden_size, 1), name='last_hidden_state')(score_first_part_t)
    # score = concatenate([score_first_part, h_t])
    attention_weights = Activation('softmax', name='attention_weight')(score)
    # if SINGLE_ATTENTION_VECTOR:
    #     a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
    #     a = RepeatVector(hidden_size)(a)
    # (batch_size, hidden_size, time_steps) dot (batch_size, time_steps, 1) => (batch_size, hidden_size, 1)
    context_vector = dot_pro([attention_weights,hidden_state_t])
    #context_vector = Reshape((hidden_size,))(context_vector)

    return context_vector
#--------------------------------------------------------------------------------
inputs = Input(shape=(32, 128, 1))
# convolution layer with kernel size (3,3)
conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
# poolig layer with kernel size (2,2)
pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)
conv_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool_1)
pool_2 = MaxPool2D(pool_size=(2, 1))(conv_2)# we remove the strides here
conv_3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool_2)
conv_4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv_3)
# poolig layer with kernel size (2,1)
pool_3 = MaxPool2D(pool_size=(2, 1))(conv_4)
conv_5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool_3)
# Batch normalization layer
batch_norm_5 = BatchNormalization()(conv_5)
conv_6 = Conv2D(512, (3, 3), activation='relu', padding='same')(batch_norm_5)
batch_norm_6 = BatchNormalization()(conv_6)
pool_4 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)
conv_7 = Conv2D(512, (2, 2), activation='relu')(pool_4)# the output here, called the time step, should be at least greater than the maximum input length of the GT.
squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)
# it converts the output of conv_7 from shape of(?,1,32,512) to (?,63,512)[ sample, timesteps, features]

# bidirectional LSTM layers with units=128
blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.25))(squeezed)
blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.25))(blstm_1)
context=attention(blstm_2)
outputs = Dense(num_class + 1, activation='softmax')(context)

#outputs = Dense(num_class + 1, activation='softmax')(blstm_2)

Attention_test_model_sent = Model(inputs, outputs)
labels = Input(name='the_labels', shape=[32], dtype='float32')# 32 is the max size of text length
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args

    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, labels, input_length, label_length])

model = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

filepath = "sentence_best_model1_attention_fine.hdf5"
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [checkpoint]

# training_img = np.array(X_train)
# train_input_length = np.array(x_train_len)
# train_label_length = np.array(y_train_len)
#note;;;att_model

hist=model.fit(x=[X_train, y_train, x_train_len, y_train_len], y=np.zeros(len(y_train)),
          batch_size=128, epochs=10,validation_data=([X_val, response to reviewers, x_val_len, y_val_len], [np.zeros(len(y_val))]), verbose = 1, callbacks = callbacks_list)


#saving and loading model
#-------------------------------------------------------------------------------------------------
test_model_sent.save('test_model_sent1.hdf5')
g=load_model('test_model_sent.hdf5')
g.summary()
g.predict(X_testp[:10])

# =======================================================================================================================

#++++++++++++++++++++++++++++++++++++++=
#not: when we train the model model both act_model and model wieths are updated.(same weigth for both model(i.e share their weigth)
y_pred=test_model_sent.predict(X_testp)
#predmodel=ctc_convolutional_func()
#y_pred=predmodel.predict(X_test)
print y_pred

out = K.get_value(K.ctc_decode(y_pred[:, :, :], input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1])[0][0])
#[:,: maxlen]


for i in range(10):
    # print the 10 first predictions
    print("Prediction :", [j for j in out[i] if j != -1], " -- Label : ", y_test[i])


#===========edit distance exampl====================================

#true=['abebe','belay','almaze']
#pred=['abebee','belau','almaaze']
true=[]# to stor value of character by removing zero which was padded previously and also this is the value of newline in the test label
for i in range(len(y_testp)):
    x=[j for j in y_testp[i] if j!=0]
    true.append(x)
pred=[]# to stor the pdicted charcter except zerro and -1 which are padded value nad blank space predicted during testing
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

''' #to save as np file
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size=0.05)
X_val=X_train[t:]
y_val=y_train[t:]
X_trainv=X_train[:t]
y_trainv=y_train[:t]

imp=np.load('/home/belay/PycharmProjects/2nd_round/icdar_dataset/all_dataset/X_trainp.npy')'''


# transfer learning for attention( in thios case we use the cnn-lstm laywer of MDPi paper and the we tarin the uper layer only.
# layers up to 16 will be the layer of the previous model.
# first we laod the MDPI paper  actial test model and the we freez the layers up to 16
g=load_model('test_model_sent.hdf5') # loading the layer of the the previused saved model so as to use its weight as transefer learning
g.summary()
x=g.get_layer('bidirectional_6').output # it is the 16th layer of MDPI paper model output
context=attention(x) # we pass x for the attention layer
#context.shape
#Out[121]: TensorShape([Dimension(None), Dimension(63), Dimension(256)])
outputs = Dense(num_class + 1, activation='softmax')(context)
att_model = Model(inputs=g.input, outputs=outputs) # it share the weight of the newly trained model

labels = Input(name='the_labels', shape=[32], dtype='float32')# 32 is the max size of text length
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([outputs, labels, input_length, label_length])

model = Model(inputs=[g.input, labels, input_length, label_length], outputs=loss_out)

for layer in model.layers[:16]: # this is used to freeze the lower layers of the MDPI paper up to 16th
    layer.trainable = False

for layer in model.layers[16:]: # used to train the layers above 16
    layer.trainable = True
# compiling the model as new model is mandatory
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')
filepath = "sentence_best_model1_attention_fine.hdf5"
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [checkpoint]
hist = model.fit(x=[X_train, y_train, x_train_len, y_train_len], y=np.zeros(len(y_train)),
                 batch_size=128, epochs=10,
                 validation_data=([X_val, y_val, x_val_len, y_val_len], [np.zeros(len(y_val))]), verbose=1,
                 callbacks=callbacks_list)
# to check charcters and words

train = []
for i in range(len(train_y_t)):
    x = [j for j in train_y_t[i] if j != 0]
    train.append(x)

trainlist = []
for i in range(len(val_y)):
    x = [j for j in train[i] if j != 0]
    trainlist.append(x)

c=[]
for i in trainlist:
    for j in i:
        c.append(j)
# to check words

size = len(c)
idx_list = [idx + 1 for idx, val in
            enumerate(c) if val == 1]
res = [c[i: j] for i, j in
       zip([0] + idx_list, idx_list +
           ([size] if idx_list[-1] != size else []))]
n=np.unique(res)
len(n)
check =  all(item in  res for item in n) #to check all elemets in n are found in res
if check is True:
    print "found"
else: print "notfound"

differnce = [item for item in resvg if item not in res]# resvg is test data label of VG  and res is all labels of training data
# 81367 words in the training set in the training set
#25596 unique words in the training set
