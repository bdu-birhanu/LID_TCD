from keras.layers import LSTM,Bidirectional,Input,Dense,Activation,Dropout,Flatten,Dot,Embedding
from keras.models import Model
from keras.optimizers import adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from preprocess import xdata,ydata,ydata2, maxlen,no_unique_words
import matplotlib.pyplot as plt
import warnings #just to remove warnings
# input data called from the preprocessor module

x_train, x_test, y_train, y_test,y_train2, y_test2 = train_test_split(xdata, ydata,ydata2, test_size=0.1)

#Function for attention layer(input---> all hidden layers, output---> context_vector)
dot_prod = Dot(axes = 1)
def attention(lstm_out):
    '''
    :param lstm_out:  all hidden layer information from the LSTM layers
    :return: context vector---> an aggregated information
    '''
    hidden_state = lstm_out
    score1 = Dense(64, activation='relu', name='attention_score_vec1')(hidden_state)
    score2 = Dense(32, activation='tanh', name='attention_score_vec')(score1)
    attention_weights = Activation('softmax', name='attention_weight')(score2)
    context_vector = dot_prod([attention_weights,hidden_state])
    return context_vector

#main model( input--> input data, output--model)
def train_model(batch_size=16, num_class=6, class2=5,maxlen=maxlen,emb_dim=32):
    '''

    :param num_class:  the types of language in the document
    :return: trained model
    '''
    input_data = Input(shape=(maxlen, ))
    embedding_layer = Embedding(len(no_unique_words)+1,
                            emb_dim,
                            input_length=maxlen)(input_data)
    lstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.25))(embedding_layer)
    lstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.25))(lstm_1)
    context=attention(lstm_2)
    fc = Flatten()(context) #flatten and pass on to the Dense output layer.
    #fc = Dense(2048, activation='softmax')(fc)
    outputs = Dense(num_class, activation='softmax')(fc)
    outputs2 = Dense(class2, activation='softmax')(fc)

    model=Model(inputs=input_data, outputs=[outputs,outputs2])

   #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.compile(loss=['categorical_crossentropy','categorical_crossentropy'], optimizer='adam', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    hist=model.fit(x_train, [y_train,y_train2], batch_size=batch_size,
                     epochs=10, verbose=1, validation_split=0.2, shuffle=True, callbacks=[early_stopping])
    return model,hist

if __name__=="__main__":
    m= train_model()
    model=m[0]
    hist=m[1]
    model.save('model_test.hdf5')
    model.summary()
    print(hist.history)
    epochs=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    train_loss=hist.history['loss']
    val_loss=hist.history['val_loss']
    print(train_loss)
    print(val_loss)
    plt.title('train_loss vs val_loss')
    plt.rcParams['figure.facecolor'] = 'White'
    plt.grid(True)
    plt.plot(epochs, train_loss, 'b--')
    plt.plot(epochs, val_loss, 'r--')
    plt.legend(['joint_train_loss', 'joint_val_loss'], loc='upper right')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')
    plt.show()


