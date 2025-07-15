import pandas as pd
from abc import ABC, abstractmethod
import logging
import numpy as np
import ast


# functions to be used in feature engineering:

# to convert the string of list to only list:
def convert(obj):
    list = []
    for i  in ast.literal_eval(obj):
        list.append(i['name'])
    return list

 # "For the 'cast' data, keep only the first 3 actor dictionaries since they correspond to the main starring roles."
# And from those dictionaries take out only the real names of actors and make list of those 3 names
def convert3(obj):
    list = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            list.append(i['name'])
            counter+=1
        else:
            break
    return list

def fetch_director(obj):
    list = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            list.append(i['name'])
            break
    return list


class DataStrategy(ABC):
    """
    Abstract base class for defining data cleaning strategies.
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        pass


class DataPreprocessStrategy(DataStrategy):
    """
    Concrete implementation of DataStrategy for preprocessing data.
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:

       try:
           # take only the required columns
            data = data[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']].copy()
            data.dropna(inplace =True)
            data.drop_duplicates(inplace=True)
            return data
       except Exception as e:
            logging.error(f"Error occurred while preprocessing data: {e}")
            raise e



class FeatureEngineeringStrategy(DataStrategy):
    """
    Concrete implementation of DataStrategy for feature engineering.
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            # Remove string literal from the featurers
            data['genres'] = data['genres'].apply(convert)
            data['keywords'] = data['keywords'].apply(convert)
            # fram the cast feature, take out only the top 3 dictionaries:
            data['cast'] = data['cast'].apply(convert3)
            # from the crew feature, take out only the director
            data['crew']  = data['crew'].apply(fetch_director)
            # convert string rows of overview feature to list of words
            data['overview'] = data['overview'].apply(lambda x: x.split())
            # Strip the spaces from all the Features.
            for feature in ['genres', 'keywords', 'cast', 'crew']:
                data[feature] = data[feature].apply(lambda x: [i.replace(" ", "") for i in x])

            # concatenate all the features and make a new feature named tags.ABC
            data['tags'] = data['overview'] + data['genres'] + data['keywords'] + data['cast'] + data['crew']

            new_df = data[['movie_id', 'title', 'tags']].copy()
            # Join the list of strings and make one single string
            new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
            # Convert all text in tags feature to lowercase.ABC
            new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

            print('new_df shape: ', new_df.shape)
            return new_df

        except Exception as e:
            logging.error(f"Error occured while feature engineering: {e}")
            raise e


class DataCleaning:
    """
    Data cleaning class which preprocesses data, and does feature engineering.
    """
    def __init__(self, data:pd.DataFrame, strategy:DataStrategy) -> None:
        """Initialize the DataCleaning class with specific data and strategy.
        """
        self.df = data
        self.strategy = strategy

    def handle_data(self) -> pd.DataFrame:
        """Handle data cleaning, Feature engieering.
        """
        try:

            return self.strategy.handle_data(self.df)

        except Exception as e:
            logging.error(f"Error occurred while cleaning data: {e}")
            raise e
