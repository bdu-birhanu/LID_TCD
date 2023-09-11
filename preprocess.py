import numpy as np
import re
from sklearn.utils import shuffle
from numpy import array
import warnings
def read_dataset():
    '''
    This function loads the data from the disk to workspace, convert each character with an integer index
    and returns the following arguments.
    xtrain --> training samples
    ytrain --> labels
    maxlen --> the maximum length of  the sequence
    no_unique_words --> list of unique characters in the document
    '''
    text = open('./Dataset/data.txt', 'r').read()
    label = open('./Dataset/labelli.txt', 'r').read()
    labelsc = open('./Dataset/labelcd.txt', 'r').read()
    text=text.lower()
    lines = text.split('\n')
    #lines=re.split(r' *[\n\።\.\?!]]* *', text)
    # Just to find unique words in the document
    word = text.replace("\n", " ")# this line ignores the \n which was previously considered as a line.
    word1=word.split(' ')
    no_unique_words=sorted(list(set(word1)))


    mapping = dict((c, i) for i, c in enumerate(no_unique_words))  # map words to an integer index
    mapping1 = dict((i, c) for i, c in enumerate(no_unique_words)) #intiger to word
    sequences = list()
    for line in lines:
        if len(line)==0: continue # ignore if the line is blank
        encoded_seq = [mapping[words] for words in line.split(' ')]
        sequences.append(encoded_seq)
    x1 = [list(line) for line in sequences]
    maxlen = max((len(r)) for r in sequences)
    minlen = min((len(r)) for r in sequences)
    minseq=list()
    for r in sequences:
        if len(r)==1:
           x=[mapping1[words] for words in r]
           minseq.append(x)
    minchar=min((len(r)) for r in minseq)
    xtrain = np.asarray([np.pad(r, (0, maxlen - len(r)), 'constant', constant_values=0) for r in x1])

    # 0=Amahric ,1= Afan Oromo, 2=Tigrigna
    lang = ['afar','awi','amharic','afan-oromo', 'somali','tigrigna']
    content= ['agriculture','health','politics','religious','sport']
    digit = [0, 1,2,3,4,5]
    digit2= [0,1,2,3,4]
    labels = label.split('\n')
    label2 = labelsc.split('\n')
    y = dict(zip(lang, digit))
    y2 = dict(zip(content, digit2))
    ylabel = []
    for index in labels:
        if index == '' or index == '\n': continue
        ylabel.append(y[index])
    ytrain=np.asarray(ylabel)
#===============================
    ylabel2 = []
    for index in label2:
        if index == '' or index == '\n': continue
        ylabel2.append(y2[index])
    ytrain2=np.asarray(ylabel2)
    allword=len(word1)
    return  xtrain,ytrain,ytrain2,no_unique_words,maxlen,allword,minlen,ylabel,ylabel2,y,y2,minseq,minchar
def langcount(): 
    y= read_dataset()
    ylang=y[7]
    # Creating an empty dictionary  
    freq = {} 
    for items in ylang: 
        freq[items] = ylang.count(items) 
      
    for key, value in freq.items(): 
        print ("% s   : % d"%(key, value)) 

def topiccount(): 
    y= read_dataset()
    ycont=y[8]
    # Creating an empty dictionary  
    freq = {} 
    for items in ycont: 
        freq[items] = ycont.count(items) 
      
    for key, value in freq.items(): 
        print ("% s    : % d"%(key, value))
  

def one_hot_encode(sequencex, n_unique):
    '''
    :param sequencex: a sequence of integer index of the text lines in the document
    :param n_unique:  number of unique words in the document
    :return:  one-hot encoded sequence of a text-line
    '''
    encoding = list()
    for value in sequencex:
        vector = [0 for _ in range(n_unique)]
        vector[value] = 1
        encoding.append(vector)
    return array(encoding)


def encoded_vector():
    '''
    :return: one-hot encoded sequnces of all training sample and
    corresponding labels
  '''
    data = read_dataset()
    xtrain = data[0]
    ytrain = data[1]
    ytrain2 = data[2]
    no_unique_words = data[3]
    maxlen = data[4]
    totalword=data[5]
    minlen=data[6]
    y_onehot = np.array(one_hot_encode(ytrain,6))
    y_onehot2 = np.array(one_hot_encode(ytrain2,5))

   # xdata, ydata=shuffle(x_onehot, y_onehot)
    xdata, ydata, ydata2=shuffle(xtrain,y_onehot, y_onehot2)
    return xdata,ydata,ydata2, maxlen,no_unique_words,totalword,minlen

encode=encoded_vector()
xdata = encode[0]
ydata = encode[1]
ydata2 = encode[2]
maxlen = encode[3]
no_unique_words = encode[4]
totalword = encode[5]
minlen=encode[6]
#if __name__=="__main__":

print ("...data loading...")
data_loaded=encoded_vector()
maps=read_dataset()

print(str(len(data_loaded[0]))+ " Samples with their Corresponding labels are loaded")
print( "The maximum sequence length is = "+ str(maxlen))
print( "The number of unique words = "+ str(len(no_unique_words)))
print( "Total number of words = "+ str(totalword))
print( "The min length of a sequence= "+ str(minlen))
print( "===============The detail data================")
print("minlen words"+str(maps[11]))
print("minlen words"+str(maps[12]))
print("The number of text lines in each language :\n" )
lang=langcount()
print(maps[9])
print("Total number of sample in each topics  :\n")
topic=topiccount()
print(maps[10])

