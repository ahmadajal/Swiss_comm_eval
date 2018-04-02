import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import json
import re
import string
# fix random seed for reproducibility
np.random.seed(7)
from gensim.models import Word2Vec
import tensorflow
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Embedding, LSTM, Conv1D, Flatten, Dropout
from keras.layers.merge import Concatenate, concatenate
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
#--------------------------------
politics_data = pd.read_csv("politics_full.csv")
sports_data = pd.read_csv("sports_full.csv")
history_data = pd.read_csv("history_full.csv")
culture_data = pd.read_csv("culture_full.csv")
comp_science_data = pd.read_csv("comp_science_full.csv")
category_dict = {
    1: 'politics',
    2: 'sports',
    3: 'history',
    4: 'culture',
    5: 'computer_science'
}
translator = str.maketrans('', '', string.punctuation)
politics_data['content'] = politics_data['content'].map(lambda x: re.sub('\n', ' ', x))\
.map(lambda x: x.translate(translator))
politics_data['category'] = [1]* len(politics_data)

sports_data['content'] = sports_data['content'].map(lambda x: re.sub('\n', ' ', x))\
.map(lambda x: x.translate(translator))
sports_data['category'] = [2]* len(sports_data)

history_data['content'] = history_data['content'].map(lambda x: re.sub('\n', ' ', x))\
.map(lambda x: x.translate(translator))
history_data['category'] = [3]* len(history_data)

culture_data['content'] = culture_data['content'].map(lambda x: re.sub('\n', ' ', x))\
.map(lambda x: x.translate(translator))
culture_data['category'] = [4]* len(culture_data)

comp_science_data['content'] = comp_science_data['content'].map(lambda x: re.sub('\n', ' ', x))\
.map(lambda x: x.translate(translator))
comp_science_data['category'] = [5]* len(comp_science_data)
#----------------------
#union data frames
df_list = [politics_data, sports_data, history_data, culture_data, comp_science_data]
full_df = pd.concat(df_list)
#shuffling the rows of dataframe
full_df = full_df.sample(frac=1)
contents = list(full_df['content'])
targets = np.array(full_df['category'])
indicies_1 = [i for i,x in enumerate(targets) if x == 1]
np.random.shuffle(indicies_1)
indicies_2 = [i for i,x in enumerate(targets) if x == 2]
np.random.shuffle(indicies_2)
indicies_3 = [i for i,x in enumerate(targets) if x == 3]
np.random.shuffle(indicies_3)
indicies_4 = [i for i,x in enumerate(targets) if x == 4]
np.random.shuffle(indicies_4)
indicies_5 = [i for i,x in enumerate(targets) if x == 5]
np.random.shuffle(indicies_5)
ratio = 0.9
train_indicies = indicies_1[0:int(ratio*len(indicies_1))]\
+indicies_2[0:int(ratio*len(indicies_2))]+indicies_3[0:int(ratio*len(indicies_3))]\
+indicies_4[0:int(ratio*len(indicies_4))]+indicies_5[0:int(ratio*len(indicies_5))]
np.random.shuffle(train_indicies)
test_indicies = indicies_1[int(ratio*len(indicies_1)):]\
+indicies_2[int(ratio*len(indicies_2)):]+indicies_3[int(ratio*len(indicies_3)):]\
+indicies_4[int(ratio*len(indicies_4)):]+indicies_5[int(ratio*len(indicies_5)):]
np.random.shuffle(test_indicies)
#---------------------------------
import nltk
tokenized = [nltk.word_tokenize(text) for text in contents]
from nltk.corpus import stopwords
# punctuation = string.punctuation+'“’—.”’“--,”' # pimp the list of punctuation to remove
def rem_stop(txt,stop_words=stopwords.words("english"),lower=True):
    """
    Removes stopwords and other things from a text, inc. numbers
    :param list txt: text tokens (list of str)
    :param list stop_words: stopwords to remove (list of str)
    :param bol lower: if to lowercase
    """
    if lower:
        return [t.lower() for t in txt if t.lower() not in stop_words and not t.isdigit()]
    else:
        return [t for t in txt if t.lower() not in stop_words and not t.isdigit()]

corpus = [rem_stop(tokens) for tokens in tokenized]
#----------------------------------

MAX_SEQUENCE_LENGTH = 5000
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
sequences = tokenizer.texts_to_sequences(corpus)

# 0 is a reserved index that won't be assigned to any word
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(targets-1))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
x_train = data[train_indicies]
y_train = labels[train_indicies]
x_test = data[test_indicies]
y_test = labels[test_indicies]
#-------------------------------------
embeddings_index = {}
f = open('glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))
#------------------------------------
EMBEDDING_DIM = 100
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)

cnn_callback = ModelCheckpoint('best_cnn_model',
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='auto')
kernel_size = [3,5,7,9]
nb_filters = [100,300,500]
# dropouts = [0,0.25,0.5]
activations = ['tanh', 'relu']
best_val_acc = 0
best_kernel_size = 0
best_nb_filter = 0
best_activation = ''
for k in kernel_size:
    for n in nb_filters:
        for act in activations:
            sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
            embedded_sequences = embedding_layer(sequence_input)
            x = Conv1D(filters=n, kernel_size=k, activation=act)(embedded_sequences)
            x = MaxPooling1D(5)(x)
            x = Dropout(0.25)(x)
            x = Flatten()(x)
            x = Dense(128, activation='relu')(x)
            preds = Dense(len(category_dict), activation='softmax')(x)

            model = Model(sequence_input, preds)
            model.compile(loss='categorical_crossentropy',
                          optimizer='rmsprop',
                          metrics=['acc'])

            # happy learning!
            model.fit(x_train, y_train, validation_split=0.1,
                      epochs=10, batch_size=512, callbacks=[cnn_callback])
            best_model = load_model('best_cnn_model')
            if best_model.evaluate(x_test, y_test)[1] > best_val_acc:
                best_kernel_size = k
                best_nb_filter = n
                best_activation = act

print('best_params')
print(best_kernel_size, best_nb_filter)
print(best_activation)

with open('best_params.txt', 'w') as f:
    f.write('best kernel size: {} \n'.format(best_kernel_size))
    f.write('best number of filters: {} \n'.format(best_nb_filter))
    f.write('best activation: '+ best_activation )
f.close()
