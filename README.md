# LID_TCD
## Dataset for Ethiopian language identification and topic classification

- This datset consists of 22,624 texts labled for two tasks:

      - Language identification: this task is used to identify the lanaguage a give text written in.
      - Topic classification: this task is also useful to classify the topics of a given text according to its content.
   

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

