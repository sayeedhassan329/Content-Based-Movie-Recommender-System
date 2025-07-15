from zenml import step
import pandas as pd
import logging
import os
import joblib
from model.data_cleaning import (DataPreprocessStrategy,
FeatureEngineeringStrategy,
DataCleaning)

from zenml.materializers.pandas_materializer import PandasMaterializer


@step(output_materializers=PandasMaterializer)
def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Zenml step which cleans the data and returns feature engineered data
    """
    if data is None:
        raise ValueError("Input data cannot be None")
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(data)}")
    if data.empty:
        raise ValueError("Input DataFrame is empty")

    logging.info(f"Received data with shape: {data.shape}")

    try:
        preprocess_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(data, preprocess_strategy)
        preprocessed_data = data_cleaning.handle_data()

        feature_engineering = FeatureEngineeringStrategy()
        data_cleaning =  DataCleaning(preprocessed_data, feature_engineering)
        feature_engineered_data = data_cleaning.handle_data()
        print(feature_engineered_data.head())


        #dir_path = os.path.expanduser("/artifacts/cleaned_data")
        root_dir = os.getcwd()
        print(root_dir)
        dir_path = os.path.join(root_dir, 'artifacts', 'cleaned_data')
        os.makedirs(dir_path, exist_ok=True)
        filename = os.path.join(dir_path,'cleaned_data.joblib')
        joblib.dump(feature_engineered_data, filename)
        logging.info(f"cleaned_data saved to {filename}")

        return feature_engineered_data

    except Exception as e:
        logging.error(f'Error in clean_data step: {e}')
        raise e
