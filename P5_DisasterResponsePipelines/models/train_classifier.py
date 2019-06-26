# import libraries

import sys
import pandas as pd
import os
from sqlalchemy import create_engine
import re
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

import pickle

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def load_data(database_filepath):
    '''
    Loads dataset from sqlite database

    Args:
        database_filepath (str): sqlite database filepath

    Returns:
        X (pd.dataframe): data features
        Y (pd.dataframe): target variables
        category_names (list): categories names list
    '''

    engine = create_engine ('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('Messages', engine)
    X = df.message
    Y = df.drop(['id','message','original','genre'], axis=1)
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    '''
    Tokenizes given text

    Args:
        text (str): text to tokenize

    Returns:
        words (list): list of words after tokenization
    '''

    # normelize text and convert to lower case
    text = text.lower() 
    
    # remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 

    # use nltk to tokenize 
    words = word_tokenize(text)

    # remove stop words
    words = [w for w in words if w not in stopwords.words("english")]

    # lemmatize words 
    words = [WordNetLemmatizer().lemmatize(w) for w in words]
    return words


def build_model(best_estimator = 'false'):
    '''
    Builds machine learning model to predict message category

    Args:
        best_estimator (bool): flag to specify to perform GridSearch 

    Returns:
        cv: gridsearch/pipline object to fit and train model 
    '''

    # setup model pipeline 
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)), 
                     ('tfidf', TfidfTransformer()), 
                     ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    # return model based on user input: 

    if (best_estimator == 'true'):
        # hyper-parameter tuning 
        parameters = {  
        'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__min_samples_split': (2, 4),
        'clf__estimator__n_estimators': (100, 250)
        }
        cv = GridSearchCV(pipeline, param_grid=parameters, verbose=7, cv=2)
        return cv
    else: 
        return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluates model performance

    Args:
        model: model to predict and evaluate
        X_test (np.array): features test set
        Y_test (np.array): target categories test set
        category_names (list): target categories names list

    Returns:
        no returns
    ''' 
    print ('Model:')
    print (model)
    # predict using test set 
    y_pred = model.predict(X_test)

    print ('Classification Report...')    
    # print model F1 score, Precision and Recall for each category
    for col in range(0, len(category_names)):
        print ('Category: {} Report:'.format(category_names[col]))
        print (classification_report(Y_test[category_names[col]], y_pred[:, col], output_dict=True)['macro avg'])

def save_model(model, model_filepath):
    '''
    Saves model to pickel file

    Args:
        model: model to save
        model_filepath (str): model pickel filename

    Returns:
        no returns
    '''
    pickle.dump(model, open(model_filepath, 'wb'))
    #pass


def main():
    if len(sys.argv) >= 3:
        
        # get input from args: 
        database_filepath, model_filepath, is_best_model = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))

        # load database:
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        # build model: 
        print('Building model...')
        model = build_model(best_estimator=is_best_model)
        
        # train model: 
        print('Training model...')
        model.fit(X_train, Y_train)
        
        # evaluate model:
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        # save model:
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument, the filepath of the pickle file to '\
              'save the model to as the second argument, and specify'\
              'GridSearch as third argument.\n\nExample:'\
              'python train_classifier.py ../data/DisasterResponse.db classifier.pkl TRUE')

if __name__ == '__main__':
    main()