### understading tokenizer
from keras.preprocessing.text import Tokenizer

nb_words = 3
# tokenizer = Tokenizer(num_words = nb_words)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(["The sun is shining in June!",
                        "September is grey.",
                        "Life is beautiful in August.",
                        "I like it",
                        "This and other things?"])
print(tokenizer.word_index)

tokenizer.texts_to_sequences(['June is beautiful and I like it!'])

print(tokenizer.word_counts)

tokenizer.texts_to_matrix(["June is beautiful and I like it!",
                           "Like August"])

# Basic Network with Text. Lets try to create a model to identify the word 'shining' in the first sentence
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM

texts = ["The sun is shining in June!",
         "September is grey.",
         "Life is beautiful in August.",
         "I like it",
         "This and other things?"]
X = tokenizer.texts_to_matrix(texts)
y = [1,0,0,0,0]
vocab_size = len(tokenizer.word_index) + 1
model = Sequential()
model.add(Dense(10, input_dim = vocab_size))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop')
model.fit(X, y, batch_size = 200, epochs = 700, verbose = 0, validation_split = 0.2, shuffle = True)
from keras.utils.np_utils import np as np
np.round(model.predict(X))


## Understanding Keras Embedding layer
# Embedding layer can be used to convert a sparse vector to a non-sparse vector (dimensionality reduction)
# However, it doesn't model semantic relationships
# Lets consider a sparse vector [0,1,0,1,1,0,0] of dimension 7. Lets see how Embedding reduces the 
# dimensionality of this vector to create non-sparse vector

model = Sequential()
model.add(Embedding(2, 2, input_length = 7))

# The first value of the Embedding function is the range of values in the input (vocab size). 
# In the example it’s 2 because we give a binary vector as input. 
# The second value is the target dimension. 
# The third is the length of the vectors we give.

model.compile(optimizer = 'rmsprop', loss = 'mse')
model.predict(np.array([[0,1,0,1,1,0,0]]))

model.layers[0].get_weights()

# Detecting 'Shining' using Embedding and ANN
model = Sequential()
model.add(Embedding(vocab_size, 10, input_length = X.shape[1]))
model.add(Flatten())
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop')
model.fit(X, y, batch_size = 200, epochs = 700, validation_split = 0.2, shuffle = True)
np.round(model.predict(X))

# Detecting 'Shining' using Embedding and LSTM
model = Sequential()
model.add(Embedding(vocab_size, 10, input_length = X.shape[1]))
model.add(LSTM(5))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop')
model.fit(X, y, batch_size = 200, epochs = 700, validation_split = 0.2, shuffle = True)
np.round(model.predict(X))

### Using Word2Vec
embeddings_index = dict()
f = open('glove.6B/glove.6B.50d.txt', encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('loaded word vectors: ', len(embeddings_index))

embedding_dimension = 10
word_index = tokenizer.word_index

# The embedding_matrix matrix maps words to vectors in the specified embedding dimension
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dimension))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    # words not found in embedding index will be all-zeros
    embedding_matrix[i] = embedding_vector[:embedding_dimension]

embedding_matrix.shape
embedding_layer = Embedding(embedding_matrix.shape[0],
                            embedding_matrix.shape[1],
                            weights = [embedding_matrix],
                            input_length = 12)
# 12 is the length of the input sequence. We can use any length here, but do make sure that
# the training data's dimension should also be 12 or any other number we choose
# If not, need to add padding
from keras.preprocessing.sequence import pad_sequences
X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X, maxlen = 12)


## We can now create any model from here. Below is an example of ANN
model = Sequential()
model.add(embedding_layer)
model.add(Flatten())
model.add(Dense(1, activation = 'sigmoid'))
model.layers[0].trainable = False
model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop')
model.fit(X, y, batch_size = 20, epochs = 700, validation_split = 0.2, shuffle = False, verbose = 0)

model.predict(X)

# Next lets use LSTM instead of ANN
model = Sequential()
model.add(embedding_layer)
model.add(LSTM(5))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop')
model.fit(X, y, batch_size = 20, epochs = 700, validation_split = 0.2, shuffle = True, verbose = 0)

model.predict(X)