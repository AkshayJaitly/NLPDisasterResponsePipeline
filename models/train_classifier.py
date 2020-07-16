import re
import sys
import pandas as pd
import numpy as np
import pickle
import nltk

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'])

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.externals import joblib
from typing import Tuple, List
import warnings
warnings.filterwarnings("ignore")

def load_data(database_filepath: str)->Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Loads data from the specified sqlite database.
    Parameters
    database_filepath(string) : path to sqlite database.
    Returns
    pandas.DataFrame : dataframe of input feature(message)
    pandas.DataFrame : dataframe of output labels(categories)
    list: list of category names
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterMessages', engine)
    X = df['message']
    Y = df.drop(['message', 'id', 'original', 'genre'], axis=1)
    column_list = Y.columns.tolist()
    return X, Y, column_list

def tokenize(text: str)->List[str]:
    """
    Cleans, tokenizes, removes stopwords and stems the input text.
    Parameters
    text(string) : input text.
    Returns
    word_tokens : list of cleaned and stemmed tokens.
    """
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
  
    # tokenize
    word_tokens = word_tokenize(text)

    all_stop_words = stopwords.words("english")
    words = [w for w in word_tokens if w not in all_stop_words ]
    # initiate Lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Lemmatize and strip
    word_tokens = []
    for word in words:
        clean_word = lemmatizer.lemmatize(word).lower().strip()
        word_tokens.append(clean_word)

    return word_tokens


def build_model()->GridSearchCV:
    """
    Build a model using sklearn pipeline and GridCvSearch
    Parameters
    None
    Returns
    GridSearchCV object.
    """
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier())) 
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [50, 100],
        'tfidf__smooth_idf':[True, False]
    }

    # optimize 
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=2)
    return model
    
def evaluate_model(model: GridSearchCV, X_test: pd.DataFrame, Y_test: pd.DataFrame, category_names: List)->None:
    """
    Model evaluation
    Parameters
    model: pipeline model
    X_test: test data set
    Y_test: labels for test data set
    category_names: names of each category
    Returns
    Classification report per category and accuracy 
    """
    # Predict model
    Y_pred = model.predict(X_test)
    print("Classification Report per Category:\n")
    for i in range(len(category_names)):
        print("Label:", category_names[i])
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))
   
    # Calculate overall accurray by taking mean
    accuracy = (Y_pred == Y_test).mean().mean()
    
    print('Mean accuracy {0:.2f}% \n'.format(accuracy*100))
    
    
def save_model(model: GridSearchCV, model_filepath: str)-> None:
    """
    Saving pickle file
    Parameters 
    model: model to be saved 
    model_filepath: filepath of the model
    Returns
    None
    """
    with open(model_filepath, 'wb') as file:
        # Pickle using joblib
        joblib.dump(model, file)


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
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()