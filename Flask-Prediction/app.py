import re
import nltk
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.pipeline import Pipeline
from flask import Flask, render_template, request, jsonify
from flask import Flask, request, jsonify

app = Flask(__name__, static_url_path='/static')

@app.route('/')
def index():
    return render_template('index.html')

# Download NLTK resources
nltk.download("stopwords")
nltk.download('omw-1.4')
nltk.download('wordnet')


# Set up stopwords and lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer= WordNetLemmatizer()


# Read datasets
df_train = pd.read_csv('train.txt', names=['Text', 'Emotion'], sep=';')
df_val = pd.read_csv('val.txt', names=['Text', 'Emotion'], sep=';')
df_test = pd.read_csv('test.txt', names=['Text', 'Emotion'], sep=';')


#removing duplicated values
index = df_train[df_train.duplicated() == True].index
df_train.drop(index, axis = 0, inplace = True)
df_train.reset_index(inplace=True, drop = True)

#removing duplicated text
index = df_train[df_train['Text'].duplicated() == True].index
df_train.drop(index, axis = 0, inplace = True)
df_train.reset_index(inplace=True, drop = True)

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
#Splitting the text from the labels
X_train = df_train['Text']
y_train = df_train['Emotion']

X_test = df_test['Text']
y_test = df_test['Emotion']

X_val = df_val['Text']
y_val = df_val['Emotion']

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
y_val = le.transform(y_val)

#Convert the class vector (integers) to binary class matrix
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_val = to_categorical(y_val)

# Tokenize words
tokenizer = Tokenizer(oov_token='UNK')
tokenizer.fit_on_texts(pd.concat([X_train, X_test], axis=0))

#converting a single sentence to list of indexes
tokenizer.texts_to_sequences(X_train[0].split())
#convert the list of indexes into a matrix of ones and zeros (BOW)
tokenizer.texts_to_matrix(X_train[0].split())
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

# Read GloVE embeddings

path_to_glove_file = 'glove.6B.200d.txt'
num_tokens = vocabSize
embedding_dim = 200 #latent factors or features
hits = 0
misses = 0
embeddings_index = {}

# Read word vectors
#with open(path_to_glove_file) as f:
with open(path_to_glove_file, 'r', encoding='utf-8', errors='ignore') as f:

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



# Load the model from the HDF5 file
CNNandLSTMmodel = load_model('7eCNNandLSTMmodel90.h5')
#load model
with open('RFmodel92.pkl', 'rb') as file:
    RF = pickle.load(file)
with open('NBmodel40.pkl', 'rb') as file:
    NB = pickle.load(file)
# Load the model from the file
with open('KNNmodel86.pkl', 'rb') as file:
    knn = pickle.load(file)
# Load the NBvectorizer
with open('NBvectorizer_model.pkl', 'rb') as file:
    NBvectorizer = pickle.load(file)
# Load the transformed X_test_counts from the file
with open('KNNvectorizer_model.pkl', 'rb') as file:
    KNNvectorizer = pickle.load(file)


def preprocess_sentence(sentence):
    sentence = normalized_sentence(sentence)
    return sentence

def get_emotion_predictions(sentence):
    sentence = preprocess_sentence(sentence)
    
    prediction_str = ""

    
    # Random Forest model prediction
    y_pred = RF.predict([sentence])
    predicted_RFemotion = y_pred[0]
    RFprobabilities = RF.predict_proba([sentence])[0]
    RFemotion_index = list(RF.classes_).index(predicted_RFemotion)
    predicted_RFemotion_probability = RFprobabilities[RFemotion_index]
    print(f"\n Random Forest Model Prediction: {predicted_RFemotion} - Probability: {predicted_RFemotion_probability}\n")
    prediction_str += f"<br>Random Forest Model Prediction: {predicted_RFemotion} - Probability: {predicted_RFemotion_probability}<br><br>"

    # Naive Bayes model prediction
    X_singleNB = NBvectorizer.transform([sentence]).toarray()
    NBpredictions = NB.predict(X_singleNB)
    predicted_NBemotion = NBpredictions
    NBprobabilities = NB.predict_proba(X_singleNB)[0]
    NBemotion_index = list(NB.classes_).index(predicted_NBemotion)
    predicted_NBemotion_probability = NBprobabilities[NBemotion_index]
    print(f"Naive Bayes Model Prediction: {predicted_NBemotion} - Probability: {predicted_NBemotion_probability}\n")
    prediction_str += f"Naive Bayes Model Prediction: {predicted_NBemotion} - Probability: {predicted_NBemotion_probability}<br><br>"

    # KNN model prediction
    X_singleknn = KNNvectorizer.transform([sentence]).toarray()
    knnpredictions = knn.predict(X_singleknn)
    predicted_knnemotion = knnpredictions
    knnprobabilities = knn.predict_proba(X_singleknn)[0]
    knnemotion_index = list(knn.classes_).index(predicted_knnemotion)
    predicted_knnemotion_probability = knnprobabilities[knnemotion_index]
    print(f"KNN Model Prediction: {predicted_knnemotion} - Probability: {predicted_knnemotion_probability}\n")
    prediction_str += f"KNN Model Prediction: {predicted_knnemotion} - Probability: {predicted_knnemotion_probability}<br><br>"


    sentence = tokenizer.texts_to_sequences([sentence])
    sentence = pad_sequences(sentence, maxlen=229, truncating='pre')
    # CNN and LSTM model prediction
    result = le.inverse_transform(np.argmax(CNNandLSTMmodel.predict(sentence), axis=-1))[0]
    proba = np.max(CNNandLSTMmodel.predict(sentence))
    print(f"CNN and LSTM Model Prediction: {result} - Probability: {proba}\n")
    prediction_str += f"CNN and LSTM Model Prediction: {result} - Probability: {proba}<br>"

    return prediction_str

# Example usage:

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            input_sentence = request.form['sentence']
            prediction_result = get_emotion_predictions(input_sentence)
            return jsonify({'result': prediction_result})  # Return the result as a JSON response
        except Exception as e:
            # Handle any exceptions and return an error message
            return jsonify({'error': str(e)})


    
if __name__ == '__main__':
    app.run(debug=True)