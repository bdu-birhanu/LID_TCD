# NLP
## Language Identification(LI)
- This method uses two stacked LSTM networks followed by an Attention layer.
- To train and test our model, we use a small dataset consists of 600 sample texts collected from 3 Ethiopian languages (Amharic, Tigrigna, and Afan-Oromo). The first two languages use the same writing system called  Abugida, while the second (Afan-Oromo) language uses Latin alphabets.
- Text-documents are given in the [text_doc.txt] file and the corresponding labels are given in [lable.txt] file. In addition, sample training progress, model summary, and results are given in[.png] file.
- Once the text-document is loaded, then each text-line in the document is changed to one-hot character encoded.


## To run the code with Terminal use the following info.
```
# Load and Pre-process data
python preprocess.py

# Train
python train.py

# Test and results
python test.py
```
## Some issues to know
1. The test environment is
    - Python 3.5.2
    - Keras 2.3.1
    - tensorflow 2.1.0

## please have a look the attached sampel outputs and modelt raining and summay information. 

## Note that this implemetation is based on :[https://link.springer.com/chapter/10.1007/978-3-030-20912-4_18], the copy of this paper from researchgate is included in this folder.
