import numpy as np
from sklearn.utils import shuffle
from numpy import array
def read_dataset():
    '''
    This function loads the data from the disk to workspace, convert each character with an integer index
    and returns the following arguments.
    xtrain --> training samples
    ytrain --> labels
    maxlen --> the maximum length of  the sequence
    unique_chars --> list of unique characters in the document
    '''
    text = open('./text_doc.txt', 'r').read()
    label = open('./label.txt', 'r').read()
    text=text.lower()
    lines = text.split('\n')
    unique_chars = sorted(list(set(text)))
    mapping = dict((c, i) for i, c in enumerate(unique_chars))
    sequences = list()
    for line in lines:
        if len(line)==0: continue # ignore if the line is blank
        encoded_seq = [mapping[char] for char in line]
        sequences.append(encoded_seq)
    xs = [list(line) for line in sequences]
    maxlen = max((len(r)) for r in sequences)
    #padding each sequence with 0 at the end
    xtrain = np.asarray([np.pad(r, (0, maxlen - len(r)), 'constant', constant_values=0) for r in xs])

    # 0=Amahric ,1= Afan Oromo, 2=Tigrigna
    lang = ['Amharic', 'Afan_oromo', 'Tigrigna']
    digit = [0, 1, 2]
    labels = label.split('\n')
    y = dict(zip(lang, digit))
    ylabel = []
    for index in labels:
        if index == '' or index == '\n': continue
        ylabel.append(y[index])
    ytrain=np.asarray(ylabel)
    return  xtrain,ytrain,unique_chars,maxlen

def one_hot_encode(sequencex, n_unique):
    '''
    :param sequencex: a sequence of integer index of the text lines in the document
    :param n_unique:  number of unique characters in the document
    :return:  one-hot encoded sequence of a text-line
    '''
    encoding = list()
    for value in sequencex:
        vector = [0 for _ in range(n_unique)]
        vector[value] = 1
        encoding.append(vector)
    return array(encoding)

data = read_dataset()
xtrain = data[0]
ytrain = data[1]
unique_chars = data[2]
maxlen = data[3]
def encoded_vector():
    '''
    :return: one-hot encoded sequnces of all training sample and
    corresponding labels
  '''
    x_one = []
    for i in xtrain:
        j = one_hot_encode(i,len(unique_chars))
        x_one.append(j)

    x_onehot = np.array(x_one)
    y_onehot = np.array(one_hot_encode(ytrain,3))

    xdata, ydata=shuffle(x_onehot, y_onehot)
    return (xdata,ydata)
#if __name__=="__main__":

print ("...data loading...")
data_loaded=encoded_vector()

print(str(len(data_loaded[0]))+ " Samples with their Corresponding labels are loaded")
print( "The maximum sequnce length is = "+ str(maxlen))
print("one-hot encoding is completed")



