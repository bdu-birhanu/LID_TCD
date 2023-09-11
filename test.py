import numpy as np
import re
from numpy import argmax
from sklearn.metrics import accuracy_score
from train import x_train, x_test, y_train, y_test, y_train2, y_test2, no_unique_words
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import pandas as pd
import warnings
#load model
model = load_model('model_test.hdf5')
#test the model
np.save('./data_numpy/x_test.npy',x_test)
np.save('./data_numpy/y_test2.npy',y_test2)
np.save('./data_numpy/x_train.npy',x_train)
np.save('./data_numpy/y_train2.npy',y_train2)
y_pred=model.predict(x_test)
y_predl=y_pred[0]
y_predcd=y_pred[1]
pred_lable = np.argmax(y_predl, axis=1)
pred_lablesc = np.argmax(y_predcd, axis=1)
true_lable = np.argmax(y_test, axis=1)
true_lablesc = np.argmax(y_test2, axis=1)
scorel=accuracy_score(np.argmax(y_test, axis=1), pred_lable)
scorec=accuracy_score(np.argmax(y_test2, axis=1), pred_lablesc)

mapping1 = dict((i, c) for i, c in enumerate(no_unique_words))  # mapp index to character

print("-----------------------Results-------------------------------------")
print("Acuracy of language identification= "+ str(scorel))
print("Acuracy of Content Detection= "+ str(scorec))

def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]

def  result_analysis():
    '''
 returns---> sequences of texts in the document
    '''
    sample=[]
    for i in x_test:
        #p=one_hot_decode(i)
        sample.append(i)

    sequences = list()
    for line in sample:
        encoded_seq = [mapping1[word] for word in line]# if word!=0]
        sequences.append(encoded_seq)
    return sequences
if __name__=="__main__":
    # the following code helps to display the information of the test data and prediction
    sequences=result_analysis()
    text = [''.join(i) for i in sequences]
    print ( "-------------Sample test text documents----------------")
    print(*text[:10], sep='\n')# it works a python 3 sp: separation

    lang = ['afar','awi','amharic','afan-oromo', 'somali','tigrigna']
    digit = [0, 1,2,3,4,5]
    y = dict(zip(digit, lang))
    print("-----------------information on language identification---------------------")

    print("-----------------True language-lables---------------------")
    a=[]
    for j in true_lable[:10]:
        a.append(y[j])
        print (y[j])
    #print (a)

    print("----------------language-Predictions-----------------------")
    b=[]
    for k in pred_lable[:10]:
        b.append(y[k])
        print (y[k])
    
    content= ['agriculture','health','politics','religious','sport']
    digit2= [0,1,2,3,4]
    y2 = dict(zip(digit2, content))
    print("-----------------information on test content detection ---------------------")

    print("-----------------True content-lables---------------------")
    for j in true_lablesc[:10]:
        print (y2[j])

    print("----------------content-Predictions-----------------------")
    for k in pred_lablesc[:10]:
        print (y2[k])
    print("----------------Analysis of the result-----------------------")
     
    print("---------Confusion matrix language-----------------------")
    a=[]
    for j in true_lable:
        a.append(y[j])
    b=[]
    for k in pred_lable:
        b.append(y[k])
    labels = np.unique(a)
    a =  confusion_matrix(a, b, labels=labels)

    print(pd.DataFrame(a, index=labels, columns=labels))

    print("---------Confusion matrix content detection--------------")
    ac=[]
    for j in true_lablesc:
        ac.append(y2[j])
    bc=[]
    for k in pred_lablesc:
        bc.append(y2[k])
    labelsc= np.unique(ac)
    ac =  confusion_matrix(ac, bc, labels=labelsc)

    print(pd.DataFrame(ac, index=labelsc, columns=labelsc))
    print(x_train[:1])
    print(x_test[:1])

    #print(confusion_matrix(true_lablesc, pred_lablesc))
