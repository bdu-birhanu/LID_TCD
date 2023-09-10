# LID_TCD
## Dataset for Ethiopian language identification and topic classification

- This datset consists of 22,624 texts labled for two tasks:

      - Language identification: this task is used to identify the lanaguage a give text written in.
      - Topic classification: this task is also useful to classify the topics of a given text according to its content.
   


## To run the code with Terminal use the following info.
```
# Load and Pre-process data
python3 preprocess.py

# Train
python3 train.py

# Test and results
python3 test.py
```
## Some issues to know
1. The test environment is
    - Python 3.5.2
    - Keras 2.3.1
    - tensorflow 2.1.0

