
# Sentiment Analysis Classifier

## Overview
This project develops sentiment analysis classifiers using machine learning techniques in Python. The project includes implementation and optimization of several algorithms such as Naive Bayes, Random Forest, and XGBoost to improve the accuracy of sentiment classification.

## Technologies Used
- Python
- Scikit-Learn for machine learning models
- Pandas for data manipulation
- Regular Expressions for text preprocessing
- Pickle for model persistence

## Getting Started

### Prerequisites
- Python 3.12.0
- Ensure all required Python packages are installed by running the following command:
  ```bash
  pip install -r requirements.txt
  ```

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/A00476814/Sentiment-Analysis-on-Kaggle-Dataset.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Sentiment-Analysis-on-Kaggle-Dataset
   ```

### Dataset
Repository contains the training and testing datasets (`train.csv` and `test.csv`).

## Usage
To run the sentiment analysis classifier, execute the following commands:

### Training the Classifier
1. Run the training script:
   ```bash
   python classifiers.py
   ```

### Testing the Classifier
1. Run the testing script:
   ```bash
   python Rishabh_Khevaria_TestDataClassification.py
   ```

## Features
- Preprocessing text data to clean and normalize for optimal machine learning processing.
- Training classifiers using different algorithms and optimizing them using GridSearchCV and RandomizedSearchCV.
- Saving trained models to disk for later use without the need for retraining.

## Testing
The testing script loads the pre-trained models and the vectorizer, processes the testing data, and evaluates the model's performance on unseen data.

