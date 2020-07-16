import sys
import pandas as pd 
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath: str, categories_filepath: str)->pd.DataFrame:
    """
    Load  two csv files into pandas dataframes and merge.
    Parameters
    messages_filepath : location of the messages csv file
    categories_filepath : location of the categories csv file
    Returns
    pandas.DataFrame: The merged dataframe
    """
    # read csv files
    categories = pd.read_csv(categories_filepath)
    messages = pd.read_csv(messages_filepath)
    # merging the datasets
    df = pd.merge(categories, messages, on='id')
    return df
    


def clean_data(df:pd.DataFrame)->pd.DataFrame:
    """
    Process a dataframe
    Parameters
    df: The pandas.Dataframe to be processed
    Returns
    pandas.DataFrame: The processed dataframe
    """
    # create a dataframe of the 36 individual category columns
    categories = pd.DataFrame(df.categories.str.split(';', expand=True))

    # select the first row of the categories dataframe
    row = categories.loc[0, :]

    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x.split('-')[0])

    # rename the columns of categories
    categories.columns = category_colnames
    # Convert category values to just binary numbers i.e. 0 and 1 
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
        categories.head()
        
    # drop the original categories column from df
    df.drop('categories', axis=1, inplace = True)
    # concatenate the original dataframe with the categories dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df = df.drop_duplicates()
    
    return df


def save_data(df: pd.DataFrame, database_filename: str)-> None:
    """
    Save the dataframe to a Sql-lite Database
    Parameters
    df: The pandas.Dataframe to be written
    database_filename: The filename path for the database
    Returns
    None
    """
    engine = create_engine('sqlite:///'+database_filename) # https://docs.sqlalchemy.org/en/13/dialects/sqlite.html
    df.to_sql('DisasterMessages', engine, if_exists="replace", index=False) # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_sql.html


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
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