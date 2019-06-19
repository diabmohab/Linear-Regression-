#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os

try:
	os.chdir(os.path.join(os.getcwd(), 'Word2vec-from-scratch-master'))
	print(os.getcwd())
except:
	pass

#  Word2Vec from Scratch using Keras
# SECTION TextPreprocessing
#%% ANCHOR Import Libiraries
import numpy as np
import tensorflow as tf
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
# Keras Word Embedding
import keras as k
from keras.preprocessing.sequence import pad_sequences

# # Load Courps
#%% ANCHOR Tokenization
# docs = 'He is the king . The king is royal . She is the royal queen'
docs = ['Well done royal king!',
        'Good work royal queen',
        'Great effort from king and queen',
        'nice work royal king',
        'Excellent royal queen !',
        'very Weak king and queen',
        'Poor effort king and queen!',
        'not good king and queen',
        'poor work king and queen',
        'Could have done better.',
        'He is a good king',
        'The good king is royal',
        'She is a bad royal queen',
        'The bad queen is royal',
        ]
# define class labels
labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0])
#%%
cleaned_doc = []
for sent in docs:
    sent = re.sub(r'[^a-zA-Z]', ' ', sent.lower())
    cleaned_doc.append(sent)

tokens = nltk.tokenize.word_tokenize(' '.join(cleaned_doc))
vocs = nltk.text.Text(tokens)

word2int = {}
int2word = {}
word2int['<PAD>'] = 0
# word2int['<SOS>'] = 1
# word2int['<EOS>'] = 2
# word2int['<UNK>'] = 3
int2word[0] = '<PAD>'
# int2word[1] = '<SOS>'
# int2word[2] = '<EOS>'
# int2word[3] = '<UNK>'
predefined_length = len(int2word)
vocab_size = len(vocs) + predefined_length
for i, word in enumerate(vocs.tokens):
    word2int[word] = i + predefined_length
    int2word[i+predefined_length] = word

def word_count(word):
    return vocs.count(word)

def one_hot(word):
    ohe = np.zeros(vocab_size)
    ohe[word2int[word]] = 1
    return ohe

def sent2padseq(sent, max_len):
    seq = [0] * max_len
    for i, word in enumerate(sent.split()):
        seq[i] = word2int[word]
    return seq

print(vocs.vocab())
print(word2int['king'])
print(int2word[word2int['king']])
print(word_count('king'))
print(one_hot('king'))
print(sent2padseq('king king', 4))
#%% ANCHOR Windowing skipgram
data = [] #Skipgram data
WINDOW_SIZE = 2
for sentence in cleaned_doc:
    sentence = sentence.split()
    for word_index, word in enumerate(sentence):
        for nb_word in sentence[max(word_index - WINDOW_SIZE, 0): min(word_index + WINDOW_SIZE, len(sentence)) + 1]:
            if nb_word != word:
                data.append([word, nb_word])

#%% Input Output Encoding
x_train = [] 
y_train = [] 
for data_word in data:
    x_train.append(one_hot(data_word[0]))
    y_train.append(one_hot(data_word[1]))

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

print(x_train.shape, y_train.shape)
print(x_train[:5,:])
print(y_train[:5, :]) 
# !SECTION
# SECTION Training
#%% Model Training Keras
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(5, input_dim=x_train.shape[1]))
model.add(tf.keras.layers.Dense(y_train.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])


#%%
history = model.fit(
    x_train,
    y_train,
    epochs=1000,
    shuffle=True,
    verbose=1,
)


#%%  
weights_list = model.get_weights()
len(weights_list)

for i in range(len(weights_list)):
    print(weights_list[i].shape)

print(word2int['queen'])

vectors = weights_list[0]
print(vectors[ word2int['queen'] ])


#%% ANCHOR get_distance
def euclidean_dist(vec1, vec2):
    return np.sqrt(np.sum((vec1-vec2)**2))

def find_closest(word_index, vectors):
    min_dist = 10000 # to act like positive infinity
    min_index = -1
    query_vector = vectors[word_index]
    for index, vector in enumerate(vectors):
        if euclidean_dist(vector, query_vector) < min_dist and not np.array_equal(vector, query_vector):
            min_dist = euclidean_dist(vector, query_vector)
            min_index = index
    return min_index

print(int2word[find_closest(word2int['queen'], vectors)])
# !SECTION
# SECTION Word Embedding

#%%
# integer encode the documents
max_length = 6
padded_docs = [sent2padseq(sent, max_length) for sent in cleaned_doc]
x_train = np.array(padded_docs)
print(x_train.shape)


#%%
# define the model
model = k.models.Sequential()
model.add(k.layers.Embedding(vocab_size, 10, input_length=max_length))
model.add(k.layers.Flatten())
model.add(k.layers.Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())


#%%
# fit the model
model.fit(x_train, labels, epochs=50, verbose=1)

#%%
# evaluate the model
loss, accuracy = model.evaluate(x_train, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))


#%%
w2v = model.layers[0].get_weights()[0]
print(w2v.shape)
king_vec = w2v[word2int['king']]
print(king_vec)


#%%
# print(euclidean_dist(w2v[0], w2v[1]))
print(int2word[find_closest(word2int['queen'], w2v)])
