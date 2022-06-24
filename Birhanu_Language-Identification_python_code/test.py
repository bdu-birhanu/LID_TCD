import numpy as np
from numpy import argmax
from sklearn.metrics import accuracy_score
from train import x_test,y_test,unique_chars
from keras.models import load_model
#load model
model = load_model('model_test.hdf5')
#test the model
y_pred=model.predict(x_test)
pred_lable = np.argmax(y_pred, axis=1)
true_lable = np.argmax(y_test, axis=1)
score=accuracy_score(np.argmax(y_test, axis=1), pred_lable)
mapping1 = dict((i, c) for i, c in enumerate(unique_chars))  # mapp index to character
print("-----------------------Results-------------------------------------")
print("Acuracy = "+ str(score))

def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]

def  result_analysis():
    '''
 returns---> sequences of texts in the document
    '''
    sample=[]
    for i in x_test:
        p=one_hot_decode(i)
        sample.append(p)

    sequences = list()
    for line in sample:
        encoded_seq = [mapping1[char] for char in line if char!=0]
        sequences.append(encoded_seq)
    return sequences
if __name__=="__main__":
    # the following code helps to display the information of the test data and prediction
    sequences=result_analysis()
    text = [''.join(i) for i in sequences]
    print ( "-------------Sample test text documents----------------")
    print(*text[:10], sep='\n')# it works a python 3

    lang = ['Amharic', 'Afan_oromo', 'Tigrigna']
    digit = [0, 1, 2]
    y = dict(zip(digit, lang))

    print("-----------------True lables---------------------")
    for j in true_lable[:10]:
        print (y[j])

    print("----------------Predictions-----------------------")
    for k in pred_lable[:10]:
        print (y[k])
