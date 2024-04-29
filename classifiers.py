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
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load the dataset
file_path = 'train.csv'
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Remove duplicate rows based on 'textID'
data.drop_duplicates(subset='textID', inplace=True)

# Remove rows where 'selected_text' or 'sentiment' is null
data.dropna(subset=['text', 'sentiment'], inplace=True)

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

    #Normalize censored words
    text = re.sub(r'\b\w*[*@#$%^&!]+\w*\b', '[CENSORED]', text)

    # Remove punctuation and numbers, keep alphabets and spaces
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Apply the cleaning function to the text column
data['cleaned_text'] = data['text'].apply(clean_text)

# Initialize CountVectorizer with English stopwords
vectorizer = CountVectorizer(stop_words='english')

# Fit and transform the cleaned text to a Bag of Words model
bow_model = vectorizer.fit_transform(data['cleaned_text'])

# Initialize TfidfTransformer
tfidf_transformer = TfidfTransformer()

# Fit and transform the Bag of Words model to TF-IDF
tfidf_model = tfidf_transformer.fit_transform(bow_model)

print("TF-IDF model shape:", tfidf_model.shape)

# Prepare the features (TF-IDF scores) and target (sentiment)
X = tfidf_model
y = data['sentiment']

# Save the vectorizer and transformer to files
vectorizer_file = 'vectorizer.pkl'
with open(vectorizer_file, 'wb') as file:
    pickle.dump(vectorizer, file)

transformer_file = 'tfidf_transformer.pkl'
with open(transformer_file, 'wb') as file:
    pickle.dump(tfidf_transformer, file)

print(f"Vectorizer saved to {vectorizer_file}")
print(f"Tfidf Transformer saved to {transformer_file}")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#--Naive Bayes classifier--#

# Initialize and train a Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Predict the sentiment on the test set
y_pred = nb_classifier.predict(X_test)

# Calculate and print the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Naive Bayes Model Accuracy: {accuracy:.4f}')
print("Naive Bayes Model Report:\n", classification_report(y_test, y_pred))

#Fine Tuning Naive Bayes classifier

# Set the parameters by cross-validation
tuned_parameters = {'alpha': [1e-3, 1e-2, 1e-1, 1, 10, 50, 100]}

# Perform grid search with cross-validation
clf = GridSearchCV(MultinomialNB(), tuned_parameters, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
clf.fit(X_train, y_train)

print("Best parameters set found on development set:")
print(clf.best_params_)
print()

# Using the best found parameters to re-train the model
nb_classifier = MultinomialNB(alpha=clf.best_params_['alpha'])
nb_classifier.fit(X_train, y_train)

# Predict the sentiment on the test set using the tuned classifier
y_pred_tuned = nb_classifier.predict(X_test)

# Calculate and print the accuracy of the tuned model
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
print(f'Tuned Naive Bayes Model Accuracy: {accuracy_tuned:.4f}')
print("Naive Bayes Model Report:\n", classification_report(y_test, y_pred_tuned))

classifier_file = 'naive_bayes_classifier.pkl'

# Save the classifier to a file
with open(classifier_file, 'wb') as file:
    pickle.dump(nb_classifier, file)

print(f"Classifier saved to {classifier_file}")

#--Random Forest Classifier--#

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)

# Predict on the test data
y_pred_rf = rf_classifier.predict(X_test)

# Calculate and print the accuracy of the Random Forest model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Random Forest Model Accuracy: {accuracy_rf:.4f}')
print("Random Forest Model Report:\n", classification_report(y_test, y_pred_rf))

#Fine Tuning Random Forest Classifier

# Define the parameter grid for Random Forest
param_grid = {
     'n_estimators': [50, 100, 200],
     'max_depth': [None, 10, 20, 30],
     'min_samples_split': [2, 5, 10]
 }

# Initialize a GridSearchCV with the Random Forest Classifier and the parameter grid
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Print the best parameters found and the corresponding score
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Retrieve the best Random Forest model from grid search
best_rf = grid_search.best_estimator_

# Predict on the test set using the best model
y_pred_rf = best_rf.predict(X_test)

# Calculate and print the accuracy of the model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Optimized Random Forest Model Accuracy: {accuracy_rf:.4f}')
print("Optimized Random Forest Model Report:\n", classification_report(y_test, y_pred_rf))

classifier_file = 'random_forest_classifier.pkl'

# Save the classifier to a file
with open(classifier_file, 'wb') as file:
    pickle.dump(best_rf, file)

print(f"Classifier saved to {classifier_file}")

#--XGBoost Classifier--#

# Initialize the label encoder
label_encoder = LabelEncoder()

# Encode the sentiment labels
y_encoded = label_encoder.fit_transform(data['sentiment'])

# Update the target dataset for training and testing
X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Now train the XGBoost classifier with the encoded labels
xgb_classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_classifier.fit(X_train, y_train_encoded)

# Predict the sentiment on the test set
y_pred_encoded = xgb_classifier.predict(X_test)

# Decode the predictions back to the original labels for accuracy calculation
y_pred = label_encoder.inverse_transform(y_pred_encoded)

# Calculate and print the accuracy of the XGBoost model
accuracy_xgb = accuracy_score(y_test, y_pred)
print(f'XGBoost Model Accuracy: {accuracy_xgb:.4f}')
print("XGBoost Model Report:\n", classification_report(y_test, y_pred))

#Fine Tuning XGBoost Classifier

# Define the parameter grid to search
param_grid = {
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 5, 7, 9],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.3],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'n_estimators': [100, 200, 300],
}

# Initialize the XGBoost classifier
xgb_classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=xgb_classifier,
    param_distributions=param_grid,
    n_iter=100,
    scoring='accuracy',
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Fit RandomizedSearchCV to the data
random_search.fit(X_train, y_train_encoded)

# Print the best parameters found and the corresponding score
print(f"Best parameters found: {random_search.best_params_}")
print(f"Best cross-validation score: {random_search.best_score_:.4f}")

# Retrieve the best XGBoost model from random search
best_xgb = random_search.best_estimator_

# Predict on the test set using the best model
y_pred_xgb = best_xgb.predict(X_test)

# Decode the predictions back to the original labels for accuracy calculation
y_pred_xgb = label_encoder.inverse_transform(y_pred_encoded)

# Calculate and print the accuracy of the model
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f'Optimized XGBoost Model Accuracy: {accuracy_xgb:.4f}')
print("Optimized XGBoost Model Report:\n", classification_report(y_test, y_pred_xgb))

classifier_file = 'xgb_classifier.pkl'

# Save the classifier to a file
with open(classifier_file, 'wb') as file:
    pickle.dump(best_xgb, file)

print(f"Classifier saved to {classifier_file}")
