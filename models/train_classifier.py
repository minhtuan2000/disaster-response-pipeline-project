import sys
import pandas as pd
import numpy as np
import re
import nltk
import pickle
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import GridSearchCV

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def load_data(database_filepath):
    """Load processed dataset from a SQL database
    
    Args:
        database_filepath: the path to the SQL database
    Returns:
        DataFrame: the processed dataset
    """ 
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('Data', engine)
    X = df['message']
    Y = df.drop(columns=['message','original','genre'])
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    """Tokenize a message
    1. Replace non-alphabetical letter and non-digit by space
    2. Tokenize words in the message
    3. Remove stop words
    4. Lemmatize word tokens
    
    Args:
        text: a text message
    Returns:
        List: the token list
    """ 
    text = re.sub(r'\W', ' ', text.lower())
    tokens = word_tokenize(text)
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        if tok not in stop_words:
            clean_tok = lemmatizer.lemmatize(tok).strip()
            clean_tokens.append(clean_tok)

    return clean_tokens


def average_accuracy(y_true, y_pred):
    """Custom scorer calculates the average accuracy over all categories"""
    y_true = np.array(y_true).T
    y_pred = np.array(y_pred).T
    accuracies = []
    for i in range(len(y_true)):
        accuracies.append(accuracy_score(y_true[i], y_pred[i]))
    return np.mean(accuracies)


def build_model():
    """Build a pipeline with GridSearchCV to train the dataset""" 

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs=-1))
    ])

    parameters = {
        'tfidf__norm': ['l1', 'l2'],
        'clf__estimator__max_depth': [2, 5, 10, 20],
        'clf__estimator__criterion': ['gini', 'entropy']
    }

    scorer = make_scorer(average_accuracy, greater_is_better=True)
    cv = GridSearchCV(pipeline, parameters, scoring=scorer, verbose=10)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Predict the test dataset and print out the precision, callback, f1-score and accuracy for each category
    
    Args:
        model: the trained model to evaluate
        X_test: the test set
        Y_test: the test labels
        categories_name: list of categories
    """
    
    Y_pred = model.predict(X_test)
    print('average accuracy:', average_accuracy(Y_test, Y_pred))
    print('---------------------')

    Y_test = pd.DataFrame(Y_test, columns=category_names)
    Y_pred = pd.DataFrame(Y_pred, columns=category_names)
    for category in category_names:
        print('category:', category)
        print(classification_report(Y_test[category], Y_pred[category]))
        print('---------------------')


def save_model(model, model_filepath):
    """Save the trained model
    
    Args:
        model: the trained model
        model_filepath: the path to save the trained model
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model.best_estimator_, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model.best_estimator_, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()