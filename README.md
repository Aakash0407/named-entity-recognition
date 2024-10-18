# Named Entity Recognition

## AIM

To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset
Named Entity Recognition (NER) is essential in NLP, focusing on identifying and classifying entities like persons, organizations, and locations in text. This project aims to develop an LSTM-based model to predict entity tags for each word in sentences. The dataset includes words, their POS tags, and standard NER tags such as ORGANIZATION, PERSON, LOCATION, and DATE. Accurate NER enhances applications like information extraction, question answering, and sentiment analysis

## DESIGN STEPS

### STEP 1:
Import necessary libraries like pandas, NumPy, and TensorFlow/Keras.
### STEP 2:
Read the dataset and use forward fill to handle null values.
### STEP 3:
Create lists of unique words and tags, and count the number of unique entries
### STEP 4:
Build dictionaries mapping words and tags to their corresponding index values.
### STEP 5:
Construct a model using Input, Embedding, Bidirectional LSTM, Spatial Dropout, and Time Distributed Dense layers, then compile it for training with the dataset.

## PROGRAM
### Name: Aakash P
### Register Number: 212222110001

```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras import layers
from keras.models import Model
data = pd.read_csv("ner_dataset.csv", encoding="latin1")
data.head(50)
data = data.fillna(method="ffill")
data.head(50)
print("Unique words in corpus:", data['Word'].nunique())
print("Unique tags in corpus:", data['Tag'].nunique())
words=list(data['Word'].unique())
words.append("ENDPAD")
tags=list(data['Tag'].unique())
print("Unique tags are:", tags)
num_words = len(words)
num_tags = len(tags)
num_words
getter = SentenceGetter(data)
sentences = getter.sentences
len(sentences)
sentences[0]
word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}
word2idx
plt.hist([len(s) for s in sentences], bins=50)
plt.show()
X1 = [[word2idx[w[0]] for w in s] for s in sentences]
type(X1[0])
X1[0]
max_len = 50
X = sequence.pad_sequences(maxlen=max_len,
                  sequences=X1, padding="post",
                  value=num_words-1)
X[0]
y1 = [[tag2idx[w[2]] for w in s] for s in sentences]
y = sequence.pad_sequences(maxlen=max_len,
                  sequences=y1,
                  padding="post",
                  value=tag2idx["O"])
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=1)
X_train[0]
y_train[0]
input_word = layers.Input(shape=(max_len,))
model = layers.Embedding(input_dim=num_words, output_dim=50, input_length=max_len)(input_word)
model = layers.SpatialDropout1D(0.1)(model)
model = layers.Bidirectional(layers.LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
output = layers.TimeDistributed(layers.Dense(num_tags, activation="softmax"))(model)
model = Model(input_word, output)

print('Aakash P')
print('212222110001')
model.summary()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_test,y_test),
    batch_size=32,
    epochs=3,
)
metrics = pd.DataFrame(model.history.history)
metrics.head()
print('Name:Aakash P')
print('Register Number: 212222110001')
metrics[['accuracy','val_accuracy']].plot()
print('Name:Aakash P')
print('Register Number: 212222110001')
metrics[['loss','val_loss']].plot()
i = 20
p = model.predict(np.array([X_test[i]]))
p = np.argmax(p, axis=-1)
y_true = y_test[i]
print('Name: Aakash P\nRegister Number:212222110001')
print("{:15}{:5}\t {}\n".format("Word", "True", "Pred"))
print("-" *30)
for w, true, pred in zip(X_test[i], y_true, p[0]):
    print("{:15}{}\t{}".format(words[w-1], tags[true], tags[pred]))

```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
![6 1](https://github.com/user-attachments/assets/1664a719-2425-45c0-93f7-5edceb703ecf)

### Histogram Plot
![6 2 1](https://github.com/user-attachments/assets/cff234b3-b80a-432e-9fa3-bb58beda2c2e)

### Sample Text Prediction
![6 2](https://github.com/user-attachments/assets/ea6101e0-1ce7-4f40-8b37-2c3e3fe5f4e8)
## RESULT
Thus, an LSTM-based model (bi-directional) for recognizing the named entities in the text is developed Successfully.
