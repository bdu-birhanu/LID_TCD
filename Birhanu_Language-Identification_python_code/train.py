from keras.layers import LSTM,Bidirectional,Input,Dense,Activation,Dropout,Flatten,Dot
from keras.models import Model
from keras.optimizers import adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from preprocess import encoded_vector,maxlen,unique_chars

# input data called from the preprocessor module
data=encoded_vector()
xdata = data[0]
ydata = data[1]

x_train, x_test, y_train, y_test = train_test_split(xdata, ydata, test_size=0.1)

#Function for attention layer(input---> all hidden layers, output---> context_vector)
dot_prod = Dot(axes = 1)
def attention(lstm_out):
    '''
    :param lstm_out:  all hidden layer information from the LSTM layers
    :return: context vector---> an aggregated information
    '''
    hidden_state = lstm_out
    score = Dense(61, activation='tanh', name='attention_score_vec')(hidden_state)
    attention_weights = Activation('softmax', name='attention_weight')(score)
    context_vector = dot_prod([attention_weights,hidden_state])
    return context_vector

#main model( input--> input data, output--model)
def train_model(batch_size=16, num_class=3):
    '''

    :param num_class:  the types of language in the document
    :return: trained model
    '''
    input_data = Input(name='input', shape=(maxlen, len(unique_chars)))
    lstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.25))(input_data)
    lstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.25))(lstm_1)
    context=attention(lstm_2)
    fc = Flatten()(context) #flatten and pass on to the Dense output layer.
    outputs = Dense(num_class, activation='softmax')(fc)
    model=Model(inputs=input_data, outputs=outputs)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    model.fit(x_train, y_train, batch_size=batch_size,
                     epochs=20, verbose=1, validation_split=0.2, shuffle=True, callbacks=[early_stopping])
    return model

if __name__=="__main__":
    model = train_model()
    model.save('model_test.hdf5')
    model.summary()

