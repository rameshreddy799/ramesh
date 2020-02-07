# Import libraries
# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
# Others
import nltk
import re
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

### Data Preprocessing
df = pd.read_csv('geo_nongeo_english.csv')
columns = ['query', 'geo_nongeo']
df = df[columns]
df['geo_nongeo'] = pd.get_dummies(df['geo_nongeo'])
df = df.dropna()

# Clean text data
def clean_text(text):
    # Remove punctuation
    text = text.translate(string.punctuation)
    
    # Convert words to lower text and split them
    text = text.lower().split()
    
    # Remove stop words
    stops = set(stopwords.words('english'))
    text = [w for w in text if not w in stops and len(w) >= 3]
    
    text = ' '.join(text)

    
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    
    # Stemming
# =============================================================================
#     text = text.split()
#     stemmer = SnowballStemmer('english')
#     stemmed_words = [stemmer.stem(word) for word in text]
#     text = ' '.join(stemmed_words)
# =============================================================================
    
    return text

# Apply the clean_text function on df['query']
df['query'] = df['query'].apply(lambda x: clean_text(x))

# Split into training and test sets
X = pd.DataFrame(df['query'])
y = pd.DataFrame(df['geo_nongeo'])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# =============================================================================
# # Tokenize and Create Sequence
# # Create sequence
# vocabulary_size = 50000
# tokenizer = Tokenizer(num_words = vocabulary_size)
# tokenizer.fit_on_texts(X_train['query'])
# 
# =============================================================================

## Tokenize and Create Sequence
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train['query'])


### Model Creation
# Getting embeddings from Glove
embeddings_index = dict()
f = open('glove.6B/glove.6B.300d.txt', encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

# Creating an embedding_matrix matrix that maps words to vectors in the specified embedding dimension
embedding_dimension = 50
vocabulary_size = len(tokenizer.word_index) + 1
print(vocabulary_size)
embedding_matrix = np.zeros((vocabulary_size, embedding_dimension))

for word, index in tokenizer.word_index.items():
    if index > vocabulary_size - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector[:embedding_dimension]
            
print(embedding_matrix)
print(embedding_matrix.shape)

# Create the embedding layer using the values of embedding matrix as weights
embedding_layer = Embedding(embedding_matrix.shape[0],
                            embedding_matrix.shape[1],
                            weights = [embedding_matrix],
                            input_length = 20)

# Convert texts to sequences and introduce padding to maintain dimensionality as 20
X_train = tokenizer.texts_to_sequences(X_train['query'])
X_test = tokenizer.texts_to_sequences(X_test['query'])

X_train = pad_sequences(X_train, maxlen = 20)
X_test = pad_sequences(X_test, maxlen = 20)


# Model structure
model_glove = Sequential()
model_glove.add(embedding_layer)
model_glove.add(LSTM(units = 300, return_sequences = True, input_shape = (embedding_matrix.shape[1], 1)))
model_glove.add(Dropout(0.2))
model_glove.add(LSTM(units = 300, return_sequences = True))
model_glove.add(Dropout(0.2))
model_glove.add(LSTM(units = 300, return_sequences = True))
model_glove.add(Dropout(0.2))
model_glove.add(LSTM(units = 300))
model_glove.add(Dropout(0.2))
model_glove.add(Dense(1, activation='sigmoid'))
model_glove.layers[0].trainable = False
model_glove.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
## Fit train data
history = model_glove.fit(X_train, np.array(y_train), validation_split=0.2, epochs = 20)

# Train on 62356 samples, validate on 15590 samples
# =============================================================================
# Epoch 1/20
# 62356/62356 [==============================] - 2034s 33ms/step - loss: 0.2713 - acc: 0.8902 - val_loss: 0.2483 - val_acc: 0.8985
# Epoch 2/20
# 62356/62356 [==============================] - 1629s 26ms/step - loss: 0.2403 - acc: 0.9021 - val_loss: 0.2297 - val_acc: 0.9063
# Epoch 3/20
# 62356/62356 [==============================] - 1609s 26ms/step - loss: 0.2250 - acc: 0.9084 - val_loss: 0.2270 - val_acc: 0.9073
# Epoch 4/20
# 62356/62356 [==============================] - 8048s 129ms/step - loss: 0.2136 - acc: 0.9133 - val_loss: 0.2249 - val_acc: 0.9096
# Epoch 5/20
# 62356/62356 [==============================] - 1692s 27ms/step - loss: 0.1987 - acc: 0.9192 - val_loss: 0.2322 - val_acc: 0.9104
# Epoch 6/20
# 62356/62356 [==============================] - 28357s 455ms/step - loss: 0.1845 - acc: 0.9254 - val_loss: 0.2288 - val_acc: 0.9115
# Epoch 7/20
# 62356/62356 [==============================] - 1704s 27ms/step - loss: 0.1692 - acc: 0.9321 - val_loss: 0.2424 - val_acc: 0.9062
# Epoch 8/20
# 62356/62356 [==============================] - 1875s 30ms/step - loss: 0.1539 - acc: 0.9377 - val_loss: 0.2505 - val_acc: 0.9112
# Epoch 9/20
# 62356/62356 [==============================] - 1789s 29ms/step - loss: 0.1401 - acc: 0.9449 - val_loss: 0.2651 - val_acc: 0.9068
# Epoch 10/20
# 62356/62356 [==============================] - 1672s 27ms/step - loss: 0.1257 - acc: 0.9513 - val_loss: 0.3144 - val_acc: 0.9064
# Epoch 11/20
# 62356/62356 [==============================] - 1677s 27ms/step - loss: 0.1153 - acc: 0.9550 - val_loss: 0.2950 - val_acc: 0.9053
# Epoch 12/20
# 62356/62356 [==============================] - 1745s 28ms/step - loss: 0.1040 - acc: 0.9595 - val_loss: 0.3207 - val_acc: 0.9038
# Epoch 13/20
# 62356/62356 [==============================] - 1766s 28ms/step - loss: 0.0965 - acc: 0.9631 - val_loss: 0.3509 - val_acc: 0.9028
# Epoch 14/20
# 62356/62356 [==============================] - 1718s 28ms/step - loss: 0.0908 - acc: 0.9651 - val_loss: 0.3509 - val_acc: 0.9055
# Epoch 15/20
# 62356/62356 [==============================] - 4104s 66ms/step - loss: 0.0826 - acc: 0.9685 - val_loss: 0.3949 - val_acc: 0.9049
# Epoch 16/20
# 62356/62356 [==============================] - 1743s 28ms/step - loss: 0.0784 - acc: 0.9702 - val_loss: 0.3942 - val_acc: 0.8996
# Epoch 17/20
# 62356/62356 [==============================] - 1713s 27ms/step - loss: 0.0757 - acc: 0.9715 - val_loss: 0.3725 - val_acc: 0.9030
# Epoch 18/20
# 62356/62356 [==============================] - 1633s 26ms/step - loss: 0.0715 - acc: 0.9736 - val_loss: 0.4066 - val_acc: 0.9030
# Epoch 19/20
# 62356/62356 [==============================] - 1705s 27ms/step - loss: 0.0676 - acc: 0.9752 - val_loss: 0.4197 - val_acc: 0.9031
# Epoch 20/20
# 62356/62356 [==============================] - 2031s 33ms/step - loss: 0.0668 - acc: 0.9753 - val_loss: 0.4505 - val_acc: 0.9038
# =============================================================================

# Saving the model for future use
from keras.models import model_from_json
model_json = model_glove.to_json() 
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model_glove.save_weights("model.h5")
print("Saved model to disk")

# =============================================================================
# # To load model from disk in future 
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")
#  
# # evaluate loaded model on test data
# loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# score = loaded_model.evaluate(X_test, y_test)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
# 
# 
# # Predict on test data
# # y_pred = loaded_model.predict_proba(X_test)
# 
# =============================================================================
y_pred = model_glove.predict_proba(X_test)


# Combining the output
y_pred = pd.DataFrame(y_pred)
index = y_test.index
y_pred.index = index
df_predicted = pd.concat([X_test, y_test, y_pred], axis = 1)
df_predicted.to_csv('predicted_test_output_v2.csv')

# ROC curve
from sklearn.metrics import roc_curve, auc
fpr, tpr, threshold = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)


# Plotting the ROC Curve
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# Converting y_test and y_pred to np arrays
y_test = np.array(y_test)
y_pred = np.array(y_pred)
y_pred = y_pred.astype(int)

# Computing TP, FP, TN and FN
def confusion_metrics(y_test, y_pred):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    for i in range(len(y_test)):
        if y_test[i] == y_pred[i] == 1:
            TP += 1
        elif y_test[i] == y_pred[i] == 0:
            TN += 1
        elif y_pred[i] == 0 and y_test[i] != y_pred[i]:
            FN += 1
        elif y_pred[i] == 1 and y_test[i] != y_pred[i]:
            FP += 1
    return (TP, FP, TN, FN)
            
y_pred_prob = np.array(df_predicted.iloc[:,-1]) 

# Function to create y_pred using y_pred_prob & thresholds
def create_y_pred(y_pred_prob, thresh):
    y_pred_list = []
    for i in range(len(y_pred_prob)):
        y_pred = 1 if y_pred_prob[i] > thresh else 0
        y_pred_list.append(y_pred)
    return y_pred_list

# Computing precision, recall, accuracy & F-scores for different thresholds
def confusion_parameters(y_test, y_pred_prob):
    para_df = pd.DataFrame(columns = ['Threshold', 'TP', 'TN', 'FP', 'FN', 'Recall', 'Precision', 'Accuracy', 'F-Score'], index = range(20))
    thresh = 0
    for i in range(len(para_df['Threshold'])):
        para_df['Threshold'][i] = thresh
        y_pred = create_y_pred(y_pred_prob, thresh)
        TP, FP, TN, FN = confusion_metrics(y_test, y_pred)
        para_df['TP'][i] = TP
        para_df['TN'][i] = TN
        para_df['FP'][i] = FP
        para_df['FN'][i] = FN
        para_df['Recall'][i] = TP/(TP + FN)
        para_df['Precision'][i] = TP/(TP + FP)
        para_df['Accuracy'][i] = float((TP + TN)/(TP + TN + FP + FN))
        para_df['F-Score'][i] = 2*para_df['Precision'][i]*para_df['Recall'][i]/(para_df['Precision'][i] + para_df['Recall'][i])
        thresh += 0.05
    return para_df

para_df = confusion_parameters(y_test, y_pred_prob)        
print(para_df) 
para_df.to_csv('Confusion_Matrix_Output v2.csv')       