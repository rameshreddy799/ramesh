import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding #https://medium.com/towards-data-science/deep-learning-4-embedding-layers-f9a02d55ac12
from keras.preprocessing import sequence

# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) =\
                             imdb.load_data(num_words=top_words)

# truncate and pad input sequences.
# https://stackoverflow.com/questions/42943291/what-does-keras-io-preprocessing-sequence-pad-sequences-do
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# create the model
embedding_vector_length = 32
model = Sequential()


model.add(Embedding(top_words, embedding_vector_length,\
                       input_length=max_review_length)) # Embedding is euivalent to one hot encoding with lesser dimentionality


model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy',
                 optimizer='adam',
                   metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y_train,
     validation_data=(X_test, y_test),
           nb_epoch=3, batch_size=64)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)

print("Accuracy: %.2f%%" % (scores[1]*100))

"""
Train on 25000 samples, validate on 25000 samples
Epoch 1/3
25000/25000 [==============================] - 1235s - loss: 0.5574 - acc: 0.6946 - val_loss: 0.4162 - val_acc: 0.8471
Epoch 2/3
25000/25000 [==============================] - 1271s - loss: 0.2951 - acc: 0.8801 - val_loss: 0.3193 - val_acc: 0.8729
Epoch 3/3
25000/25000 [==============================] - 1291s - loss: 0.2319 - acc: 0.9094 - val_loss: 0.3075 - val_acc: 0.8800
"""