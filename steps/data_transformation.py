from model.data_transformation import DataTransformation
from zenml import step
import pandas as pd



@step
def transform_data(data: pd.DataFrame):

    vectors = DataTransformation().transform(data)
    return vectors
