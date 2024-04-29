import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the test dataset
test_file_path = 'test.csv'
test_data = pd.read_csv(test_file_path, encoding='ISO-8859-1')

# Preprocessing function using regular expressions for basic cleaning
def clean_text(text):
    if pd.isnull(text):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Handle negations by merging negation word with following word
    text = re.sub(r'\b(not|no|n\'t)\s+(\w+)', r'\1_\2', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Normalize censored words
    text = re.sub(r'\b\w*[*@#$%^&!]+\w*\b', '[CENSORED]', text)

    # Remove punctuation and numbers, keep alphabets and spaces
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Remove duplicate rows based on 'textID'
test_data.drop_duplicates(subset='textID', inplace=True)

# Remove rows where 'selected_text' or 'sentiment' is null
test_data.dropna(subset=['text', 'sentiment'], inplace=True)

# Apply the cleaning function to the text column
test_data['cleaned_text'] = test_data['text'].apply(clean_text)

# Load the vectorizer and transformer
vectorizer_file = 'vectorizer.pkl'
with open(vectorizer_file, 'rb') as file:
    vectorizer = pickle.load(file)

transformer_file = 'tfidf_transformer.pkl'
with open(transformer_file, 'rb') as file:
    tfidf_transformer = pickle.load(file)

# Transform the preprocessed text into TF-IDF vectors
test_bow_model = vectorizer.transform(test_data['cleaned_text'])
test_tfidf_model = tfidf_transformer.transform(test_bow_model)

#--Naive Bayes classifier--#

# Load the classifier from the file
classifier_file = 'naive_bayes_classifier.pkl'

with open(classifier_file, 'rb') as file:
    nb_classifier = pickle.load(file)
print(f"Loaded classifier from {classifier_file}")

# Predict the sentiment on the test set
y_test_pred = nb_classifier.predict(test_tfidf_model)

if 'sentiment' in test_data.columns:
    y_test_true = test_data['sentiment']
    test_accuracy = accuracy_score(y_test_true, y_test_pred)
    print(f'Test Accuracy: {test_accuracy:.4f}')
else:
    print("Test predictions:")
    print(y_test_pred)

#--Random Forest classifier--#

# Load the classifier from the file
classifier_file = 'random_forest_classifier.pkl'

with open(classifier_file, 'rb') as file:
    random_forest_classifier = pickle.load(file)
print(f"Loaded classifier from {classifier_file}")

# Predict the sentiment on the test set
y_test_pred = random_forest_classifier.predict(test_tfidf_model)

if 'sentiment' in test_data.columns:
    y_test_true = test_data['sentiment']
    test_accuracy = accuracy_score(y_test_true, y_test_pred)
    print(f'Test Accuracy: {test_accuracy:.4f}')
else:
    print("Test predictions:")
    print(y_test_pred)

#--XGBoost classifier--#

# Load the classifier from the file
classifier_file = 'xgb_classifier.pkl'

with open(classifier_file, 'rb') as file:
    xgb_classifier = pickle.load(file)
print(f"Loaded classifier from {classifier_file}")

# Initialize the label encoder
label_encoder = LabelEncoder()

# Encode the sentiment labels
y_encoded = label_encoder.fit_transform(test_data['sentiment'])

# Predict the sentiment on the test set
y_test_pred = xgb_classifier.predict(test_tfidf_model)

if 'sentiment' in test_data.columns:
    y_test_true = y_encoded
    test_accuracy = accuracy_score(y_test_true, y_test_pred)
    print(f'Test Accuracy: {test_accuracy:.4f}')
else:
    print("Test predictions:")
    print(y_test_pred)