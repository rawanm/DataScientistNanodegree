# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Loads dataset and returns dataframe

    Args:
        messages_filepath (str): messages .csv dataset filepath
        categories_filepath (str) categories .csv dataset filepath

    Returns:
        df (pd.dataframe): messages and categories merged pandas dataframe 
    '''

    messages = pd.read_csv('./'+ messages_filepath)
    categories = pd.read_csv('./'+ categories_filepath)
    df = messages.merge(categories, on='id')

    return df

def clean_categories_data(categories_df): 
    '''
    Cleans categories columns and data types

    Args:
        categories_df (pd.dataframe): categories dataframe

    Returns:
        categories (pd.dataframe): cleaned categories pandas dataframe 
    '''

    categories = categories_df.str.split(';',expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    category_colnames = [col.split('-')[0] for col in row]
    categories.columns = category_colnames

    # set each value to be the last character of the string
    # convert column from string to numeric
    for column in categories:
        categories[column] = categories[column].str[-1:].astype(int)
    
    return categories

def clean_data(df):
    '''
    Cleans dataframe and removed duplicates 

    Args:
        df (pd.dataframe): merged dataframe

    Returns:
        df (pd.dataframe): cleaned merged pandas dataframe 
    '''

    # clean categories data and concat with main dataset: 
    categories = clean_categories_data(df.categories)
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat((df, categories),  axis=1)
    

    # check for duplicates: 
    print('Number of duplicates before cleaning...\n   {}'
        .format(df[df.duplicated(subset=None, keep=False)].count()[0]))

    # drop duplicates
    df.drop_duplicates(subset =None, keep =False, inplace =True)

    print('Number of duplicates after cleaning...\n   {}'
        .format(df[df.duplicated(subset=None, keep=False)].count()[0]))
    return df


def save_data(df, database_filename):
    '''
    Saves dataframe to sqlite database

    Args:
        df (pd.dataframe): merged and cleaned dataframe
        database_filename (str): sqlite database filename

    Returns:
        no returns
    '''

    # create sqlite database engine
    engine = create_engine('sqlite:///'+ database_filename)

    # save dataframe to sqllite database and replace if exists
    df.to_sql('Messages', engine, index=False, if_exists='replace')
    print ('Sample of saved data...') 
    print (engine.execute("SELECT * FROM Messages").fetchall()[0])

def main():
    if len(sys.argv) == 4:

        # get input from args: 
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        # load data: 
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        # clean data: 
        print('Cleaning data...')
        df = clean_data(df)
        
        # save data:
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()