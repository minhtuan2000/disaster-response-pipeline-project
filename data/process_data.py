import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Load messages dataset and categories dataset to a DataFrame
    
    Args:
        messages_filepath: the path to messages dataset
        categories_filepath: the path to categories dataset 
    Returns:
        DataFrame: the merged dataset
    """ 
    messages = pd.read_csv(messages_filepath, index_col='id')
    categories = pd.read_csv(categories_filepath, index_col='id')
    df = messages.merge(categories, how='outer', left_index=True, right_index=True)
    return df


def clean_data(df):
    """Split the categories column to 36 individual columns and convert their values to 0 / 1.
    Drop the original categories column.
    Drop duplicated rows.
    
    Args:
        df: the dataset
    Returns:
        DataFrame: the cleaned dataset
    """ 
    # split categories into separate category columns
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    # convert category values to 0 and 1
    for column in categories:
        categories[column] = categories[column].str.split('-', expand=True)[1]
        categories[column] = categories[column].astype(int)

    # replace categories column in df with new category columns
    df = df.drop(columns='categories')
    df = df.merge(categories, how='outer', left_index=True, right_index=True)

    # drop duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """Save the processed dataset to a SQL database.
    
    Args:
        df: the dataset
        database_filename: the path to the SQL database
    """ 
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('Data', engine, index=False)


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