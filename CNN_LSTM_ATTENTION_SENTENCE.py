from keras import backend as K
from keras.layers import  Merge, RepeatVector, Dot, Bidirectional,LSTM,Input, Dense, Activation,Lambda,Concatenate,Conv2D,MaxPool2D
from keras.optimizers import  Adam
from keras.layers import Input, Dense, Activation,BatchNormalization, Lambda,MaxPool2D
from keras.utils import  to_categorical
from keras.models import Model,load_model
from random import randint
import numpy as np
from keras.callbacks import ModelCheckpoint
from numpy import array, argmax
from sklearn.model_selection import train_test_split
input_dim=32
time_steps=32 # th output dimesion of CNN layer
n=200000
n_unique=280
max_length= 32

def get_data():
    #train_x_i = np.memmap('/home/belay/PycharmProjects/3rd_round/text_line/train128by32/X_trainp_pg_vg.npy')
    train_x_i=np.load('/home/belay/PycharmProjects/3rd_round/text_line/train128by32/X_trainp_pg_vg.npy')# [sample size,32,128] 318706
    train_y_t = np.load('/home/belay/PycharmProjects/3rd_round/text_line/train128by32/y_train_all_one_hot.npy')
    #train_y_t=np.load('/home/belay/PycharmProjects/3rd_round/text_line/train128by32/y_trainp_pg_vg.npy')
    test_x_pg = np.load('/home/belay/PycharmProjects/3rd_round/text_line/test128by32/X_test_pg.npy')
    test_y_pg =np.load('/home/belay/PycharmProjects/3rd_round/text_line/test128by32/y_test_pg.npy')
    test_xp  = np.load('/home/belay/PycharmProjects/3rd_round/text_line/test128by32/X_testp.npy')
    test_y_p=np.load('/home/belay/PycharmProjects/3rd_round/text_line/test128by32/y_testp.npy')
    test_x_vg = np.load('/home/belay/PycharmProjects/3rd_round/text_line/test128by32/X_test_vg.npy')
    test_y_vg=np.load('/home/belay/PycharmProjects/3rd_round/text_line/test128by32/y_test_vg.npy')
    #==============================================================================================================
    X_train = train_x_i.reshape(train_x_i.shape[0], train_x_i.shape[1], train_x_i.shape[2], 1) #318706


    #val_tex=np.load('/home/belay/PycharmProjects/3rd_round/word_synth/y_val.npy')
    train_x,val_x, train_y, val_y = train_test_split(train_x_i, train_y_t, test_size=0.07)
    #reshape the size of the image from 3D to 4D so as to make the input size is similar with it.
    #X_train=train_x.reshape(train_x.shape[0],train_x.shape[1],train_x.shape[2],1) #[samplesize,32,128,1]
    X_test_pg=test_x_pg.reshape(test_x_pg.shape[0],test_x_pg.shape[1],test_x_pg.shape[2],1)
    X_test_vg=test_x_vg.reshape(test_x_vg.shape[0],test_x_vg.shape[1],test_x_vg.shape[2],1)
    X_test_p=test_xp .reshape(test_xp .shape[0],test_xp .shape[1],test_xp.shape[2],1)
    X_val=val_x.reshape(val_x.shape[0],val_x.shape[1],val_x.shape[2],1)
#def get_data(n, time_steps, input_dim, n_unique):
    # x=np.random.standard_normal(size=(n, time_steps,input_dim))
    #y= np.random.randint(low=0,high=8,size=(n,length))
    train_imagei = np.load('/home/belay/PycharmProjects/3rd_round/text_line/train128by32/X_trainp_pg_vg.npy')
    #y = np.load('/home/belay/PycharmProjects/3rd_round/text_line/train128by32/y_trainp_pg_vg.npy')
    y=np.load('/home/belay/PycharmProjects/2nd_round/icdar_dataset/all_dataset/12.npy')

    x=train_imagei[:200000]
    #y=train_texi
    return x,y
# def get_datay(n,length):
#     y= np.random.randint(low=0,high=8,size=(n,length))
#     return  y
    #return [randint(0, n_unique - 1) for _ in range(length)]
def one_hot_encode(sequence, n_unique):
	encoding = list()
	for value in sequence:
		vector = [0 for _ in range(n_unique)]
		vector[value] = 1
		encoding.append(vector)
	return array(encoding)

def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]

def pair():
    #x,y= get_data(n,time_steps,input_dim,n_unique)
    x,y=get_data()
    #y=get_datay(n,length)
    k=[]
    for i in y:
        j=one_hot_encode(i,280)
        k.append(j)

    y_new=np.array(k)
    return y_new

x_train,y_train=get_data()
x_new=x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
#define laye obeject for all y stpes and propagate any keras tensor  through these layers
repeat = RepeatVector(63) # the dimension of cnn output which is 63 in this case
r= RepeatVector(1)# to repeat the vector output dimention(batch_siz,1,hiden_state) to be concatneted with the cnotext vector before feeding to the post lstm
concatenat = Concatenate(axis=-1)
fc1 = Dense(63, activation = "relu") # it takes the n*f feature and convert to n*63
fc2 = Dense(1, activation = "tanh")
activation = Activation('softmax', name='attention_weights')
dot_pro = Dot(axes = 1)


def attention(encoder,hiddenstate):
    #2  calculate alignment score
    #2.1 repeat the previous hidden state of decoderLstm  from (sample-size, hidden) to be of the shape  (sample_size, sequence_length/s_prev, hidden_state_of_decoder)
    #it embedss the timestep in bettewn the batchsize and hidden stats of the decoder
    hiddenstate = repeat(hiddenstate)
    #2.2 concatnet the  output hidden state of encoderLSTM and the previous hidden state size of decoderLstm
    concat = concatenat([encoder, hiddenstate])

    #2.3 propagate the concatenated output (which is called the alignment weight in this case) through fully connected network that has l function
    # source: https://www.tensorflow.org/tutorials/text/nmt_with_attention
    alignment = fc1(concat)
    #2.4 compute the final aligment score by multiplying the weighted matrix

    alignment_score = fc2(alignment)
    #3 compute the attention weight  using soft max activation (softmaxing the alignment score)
    atten_weight = activation(alignment_score)
    #4 compute the contect vector using dot product between the attention weight and encoderLSTM outputs
    context_vec = dot_pro([atten_weight, encoder])
    return context_vec


Hencoder = 128
Hdecoder = 128
decoder_LSTM_cell = LSTM(Hdecoder, return_state = True)
output_layer = Dense(n_unique , activation='softmax')


X = Input(shape=(32, 128, 1))
# X = Input(shape=(time_steps, input_dim))
Hstate = Input(shape=(Hdecoder,), name='s0')
Cstate = Input(shape=(Hdecoder,), name='c0')
hiddenstate = Hstate # decoder hidden state inputs
cell_state = Cstate # decoder cell state
outputs = []

#1.  produce encoder hidden state of each element in the input sequence
    # convolution layer with kernel size (3,3)

# convolution layer with kernel size (3,3)
conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(X)
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
encoder_out = Bidirectional(LSTM(128, return_sequences=True, dropout=0.25))(blstm_1) # (sampe,timstep, features)
# 1. produce hidden state for each element in the input sequence
# this is the intial hiddenstate od the decoder
#transfer learning
#=======================================================
g=load_model('/home/belay/PycharmProjects/3rd_round/Models/Sentence_CNN_LSTM_CTC/test_model_sent.hdf5')
encoder_out=g.get_layer('bidirectional_6').output


#=================================================
out_dense = output_layer(hiddenstate)# it helps to minimize the error rate of the first character

for t in range(max_length):
    context = attention(encoder_out,hiddenstate)

          # 6. The context vector, we produced will, then be concatenated with the previous decoder output ( we pass the (hidden state, cell state) of the decoderLSTM

    hid = r(out_dense)
    dec_input = concatenat([context, hid])
    states = [hiddenstate, cell_state]

    # 6. The context vector, we produced will, then be concatenated with the previous decoder output ( we pass the (hidden state, cell state) of the decoderLSTM

    hiddenstate, _, cell_state = decoder_LSTM_cell(dec_input, initial_state=states)

    # 7 the output hidden state of the decoder pass through FC to produce the final output probablity which will scored by softmax too.
    out_dense = output_layer(hiddenstate)
    # the context vector at step t computed from attention mechanism ([Dimension(None), Dimension(1), Dimension(256)])
    # hid = r(hiddenstate)
    # dec_input = dot_pro([context, hid])



    # this will feed to the decoder lstm(the context vector and the hiddwnstate output should concatneted first.
    outputs.append(out_dense)

model = Model(inputs=[X, Hstate, Cstate], outputs=outputs)
    #pmodel=Model(inputs=X,output=outputs)
model = Model(inputs=[g.input, Hstate, Cstate], outputs=outputs)
for layer in model.layers[:17]: # this is used to freeze the lower layers of the MDPI paper up to 16th
    layer.trainable = False

for layer in model.layers[17:]: # used to train the layers above 16
    layer.trainable = True



#model = model(max_length, Hencoder, Hdecoder)

model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                    metrics=['accuracy'])


Hstaten = np.zeros((len(X_train), Hdecoder))
Cstaten = np.zeros((len(X_train), Hdecoder))
# y_train=np.array(y)
outputsn = list(train_y_t.swapaxes(0,1))
#outputs=y
checkpoint = ModelCheckpoint('model_30rms{epoch:01d}.hdf5', period=1)


hist=model.fit([X_train, Hstaten, Cstaten], outputsn, epochs=25, batch_size=128, validation_split=0.07,callbacks=[checkpoint])

x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)


Hstaten = np.zeros((318706, Hdecoder))
Cstaten = np.zeros((318706, Hdecoder))

p = model.predict([x_train[:5], Hstaten, Cstaten]) # outputs...> (max-len-chars, sample_size, len-unique) 32,5,280
p1=(np.array(p).swapaxes(0,1))
#(sample_size, max-len-chars, len_unique) (5,32,280)

y_pre=[]
for i in range(len(test_y_pg)):
    pred=one_hot_decode(p1[i])
    y_pre.append(pred)
y_pred=np.array(y_pre)

true=[]# to stor value of charcter by removing zeero which was padded prvieously and also this is the value of newline in the test label
for i in range(len(y_test)):
    x=[j for j in y_test[i] if j!=0]
    true.append(x)
pred=[]# to store the predicted charcter except zerro and -1 which are padded value nad blach soace predicted during testing
for i in range(len(y_pred)):
    x=[j for j in y_pred[i] if j not in(0,-1)]
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
cerp=(float(err)/totalchar)*100



#prediction = np.argmax(prediction, axis=-1)
imt=np.append(im1,im2,axis=0)

epochs=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
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

