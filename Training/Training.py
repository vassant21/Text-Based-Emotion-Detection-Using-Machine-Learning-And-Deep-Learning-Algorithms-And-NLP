

import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import re
import nltk
import pickle
#import string
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LSTM, Embedding
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
#from sklearn.metrics import f1_score
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Reshape
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
#import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Reshape
from tensorflow.keras.callbacks import ModelCheckpoint



# Download NLTK resources
nltk.download("stopwords")
nltk.download('omw-1.4')
nltk.download('wordnet')



# Set up stopwords and lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer= WordNetLemmatizer()


# Read datasets
df_train = pd.read_csv('/content/train.txt', names=['Text', 'Emotion'], sep=';')
df_val = pd.read_csv('/content/val.txt', names=['Text', 'Emotion'], sep=';')
df_test = pd.read_csv('/content/test.txt', names=['Text', 'Emotion'], sep=';')

# Display data information
print("Training data shape:", df_train.shape)
print("Testing data shape:", df_test.shape)
print("Validation data shape:", df_val.shape)


#removing duplicated values
index = df_train[df_train.duplicated() == True].index
df_train.drop(index, axis = 0, inplace = True)
df_train.reset_index(inplace=True, drop = True)

#removing duplicated text
index = df_train[df_train['Text'].duplicated() == True].index
df_train.drop(index, axis = 0, inplace = True)
df_train.reset_index(inplace=True, drop = True)

#removing duplicated text
index = df_val[df_val['Text'].duplicated() == True].index
df_val.drop(index, axis = 0, inplace = True)
df_val.reset_index(inplace=True, drop = True)


# %%
# Bar Plot - Class Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Emotion', data=df_train)
plt.title('Class Distribution - Training Data')
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.savefig('classDistribution-training.png')
plt.show()

# Bar Plot - Class Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Emotion', data=df_test)
plt.title('Class Distribution - testing Data')
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.savefig('classDistribution-testing.png')
plt.show()

# Bar Plot - Class Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Emotion', data=df_val)
plt.title('Class Distribution - validation Data')
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.savefig('classDistribution-validation.png')
plt.show()

"""check that there is no data leakage"""



def lemmatization(text):
    lemmatizer= WordNetLemmatizer()

    text = text.split()

    text=[lemmatizer.lemmatize(y) for y in text]

    return " " .join(text)

def remove_stop_words(text):

    Text=[i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

def Removing_numbers(text):
    text=''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text):

    text = text.split()

    text=[y.lower() for y in text]

    return " " .join(text)

# %%
def Removing_punctuations(text):
    ## Remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛',"", )

    ## remove extra whitespace
    text = re.sub('\s+', ' ', text)
    text =  " ".join(text.split())
    return text.strip()



def Removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)



def remove_small_sentences(df):
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan



def normalize_text(df):
    df.Text=df.Text.apply(lambda text : lower_case(text))
    df.Text=df.Text.apply(lambda text : remove_stop_words(text))
    df.Text=df.Text.apply(lambda text : Removing_numbers(text))
    df.Text=df.Text.apply(lambda text : Removing_punctuations(text))
    df.Text=df.Text.apply(lambda text : Removing_urls(text))
    df.Text=df.Text.apply(lambda text : lemmatization(text))
    return df



def normalized_sentence(sentence):
    sentence= lower_case(sentence)
    sentence= remove_stop_words(sentence)
    sentence= Removing_numbers(sentence)
    sentence= Removing_punctuations(sentence)
    sentence= Removing_urls(sentence)
    sentence= lemmatization(sentence)
    return sentence



df_train= normalize_text(df_train)
df_test= normalize_text(df_test)
df_val= normalize_text(df_val)



#Preprocess text
X_train = df_train['Text'].values
y_train = df_train['Emotion'].values

X_test = df_test['Text'].values
y_test = df_test['Emotion'].values

X_val = df_val['Text'].values
y_val = df_val['Emotion'].values




def train_model(model, data, targets):

    # Create a Pipeline object with a TfidfVectorizer and the given model
    text_clf = Pipeline([('vect',TfidfVectorizer()),
                         ('clf', model)])
    # Fit the model on the data and targets
    text_clf.fit(data, targets)
    return text_clf

# %%

#Train the model with the training data
RF = train_model(RandomForestClassifier(random_state = 0), X_train, y_train)


# %%
#test the model with the test data
y_pred=RF.predict(X_test)

# %%
#calculate the accuracy
RF_accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: ', RF_accuracy,'\n')

# %%

# Save the model to a file
with open('RFmodel.pkl', 'wb') as file:
    pickle.dump(RF, file)

# %%

# Create the CountVectorizer
vectorizer = CountVectorizer()

# %%

# Transform the text data into numerical features
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)


# %%
# Train the Gaussian Naive Bayes classifier
NB = GaussianNB()
NB.fit(X_train_counts.toarray(), y_train)


# %%
# Test the model with the test data
y_pred = NB.predict(X_test_counts.toarray())


# %%
# Calculate the accuracy
NB_accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: ', NB_accuracy, '\n')


# %%

# Save the model to a file
with open('NBmodel.pkl', 'wb') as file:
    pickle.dump(NB, file)

vectorizer = TfidfVectorizer()


# %%

# Transform the text data into numerical features
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# %%
# Train the model with the training data
knn = KNeighborsClassifier(n_neighbors=18)
knn.fit(X_train_tfidf, y_train)

# %%
# Test the model with the test data
y_pred = knn.predict(X_test_tfidf)
# Calculate the accuracy
knn_accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: ', knn_accuracy, '\n')


# %%
# Save the model to a file
with open('KNNmodel.pkl', 'wb') as file:
    pickle.dump(knn, file)
 # Test the model with the test data

# %%
#CNN code starts here

#Splitting the text from the labels
X_train = df_train['Text']
y_train = df_train['Emotion']

X_test = df_test['Text']
y_test = df_test['Emotion']

X_val = df_val['Text']
y_val = df_val['Emotion']


# %%
# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
y_val = le.transform(y_val)


# %%

#Convert the class vector (integers) to binary class matrix
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)


# %%
# Tokenize words
tokenizer = Tokenizer(oov_token='UNK')
tokenizer.fit_on_texts(pd.concat([X_train, X_test], axis=0))


# %%
#converting a single sentence to list of indexes
tokenizer.texts_to_sequences(X_train[0].split())


# %%
#convert the list of indexes into a matrix of ones and zeros (BOW)
tokenizer.texts_to_matrix(X_train[0].split())

# %%
#the sentence contains three words and the size of the vocabulary is 14325
tokenizer.texts_to_matrix(X_train[0].split()).shape

sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_test = tokenizer.texts_to_sequences(X_test)
sequences_val = tokenizer.texts_to_sequences(X_val)

maxlen = max([len(t) for t in df_train['Text']])
maxlen

X_train = pad_sequences(sequences_train, maxlen=229, truncating='pre')
X_test = pad_sequences(sequences_test, maxlen=229, truncating='pre')
X_val = pad_sequences(sequences_val, maxlen=229, truncating='pre')

vocabSize = len(tokenizer.index_word) + 1
print(f"Vocabulary size = {vocabSize}")


# %%
# Read GloVE embeddings

path_to_glove_file = '/content/drive/MyDrive/new/glove.6B.200d.txt'
num_tokens = vocabSize
embedding_dim = 200 #latent factors or features
hits = 0
misses = 0
embeddings_index = {}

# Read word vectors
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs
print("Found %s word vectors." % len(embeddings_index))

# Assign word vectors to our dictionary/vocabulary
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))


# %%
adam = Adam(learning_rate=0.005)
model = Sequential()
model.add(Embedding(vocabSize, 200, input_length=maxlen, weights=[embedding_matrix], trainable=False))
model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Reshape((1, 128)))  # Add Reshape layer to convert to 3D shape
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()

plot_model(model, show_shapes=True)
plt.savefig('networkLayer.png')

# %%
Define the checkpoint path and filename
checkpoint_path = "CNNLSTMmodel_checkpoint.h5"

# Define the ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    checkpoint_path,
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='min',
    save_freq='epoch'
)


# %%
history = model.fit(X_train,
                    y_train,
                    validation_data=(X_val, y_val),
                    verbose=1,
                    batch_size=256,
                    epochs=30,
                    callbacks=[callback, checkpoint_callback]
                   )

# %%
# Get the accuracy and loss values from the history object
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


# %%
# Plot the accuracy values
plt.figure(figsize=(8, 5))
plt.plot(accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.title('Accuracy over Training Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('AccuracyPlot.png')
plt.show()


# %%
# Plot the loss values
plt.figure(figsize=(8, 5))
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Loss over Training Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('LossPlot.png')
plt.show()


# %%
model.save('CNNLSTMmodel.h5')
print("Model saved.")



