import logging
import pandas as pd
from pandas.core.indexes.datetimes import prefix_mapping
from zenml import step

class IngestData:
    """
    Data ingest class which ingests data from source and returns dataframe
    """
    def __init__(self) -> None:
        """
        Initialize the IngestData class.
        """
        pass

    def get_data(self, data_path1: str, data_path2: str) -> pd.DataFrame:
        """
        Ingest data from the given path and return a pandas DataFrame.
        """
        logging.info("Ingesting data from %s", data_path1)
        data1 = pd.read_csv(data_path1  )

        logging.info("Ingesting data from %s", data_path2)
        data2 = pd.read_csv(data_path2)

        # merge dataframes
        data = pd.merge(data1, data2, on='title')

        return data


from zenml.materializers.pandas_materializer import PandasMaterializer

@step(output_materializers=PandasMaterializer)
def ingest_data(data_path1: str, data_path2: str) -> pd.DataFrame:
    """
    Ingest data from source and return dataframe.

    Args:
        data_path: Path to the data file.
    Returns:
        DataFrame: Ingested data.
    """
    try:
        ingest_data = IngestData()
        df = ingest_data.get_data(data_path1, data_path2)
        if df.empty:
            raise ValueError("Dataframe is empty")
        return df
    except Exception as e:
        logging.error(f"Error occurred while ingesting data: {e}")
        raise e


if __name__ == "__main__":
    df = ingest_data('/home/sayeed-hassan/Desktop/movie_recommender_project/movie_recommender_system/data/tmdb_5000_movies.csv',
        '/home/sayeed-hassan/Desktop/movie_recommender_project/movie_recommender_system/data/tmdb_5000_credits.csv')
    print(df.head())
