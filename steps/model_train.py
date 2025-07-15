
from sklearn.neighbors import NearestNeighbors

from model.model_dev import ModelTrain
import mlflow.sklearn
from zenml import step
import os

from materializers.custom_materializer import SKLearnMaterializer
import mlflow

import numpy as np
from mlflow.pyfunc import PythonModel
from zenml.client import Client


class NearestNeighborsWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        if not isinstance(model_input, np.ndarray):
            model_input = np.array(model_input)
        distances, indices = self.model.kneighbors(model_input)
        return indices.tolist()

experiment_tracker = Client().active_stack.experiment_tracker.name


@step#(experiment_tracker=experiment_tracker) #output_materializers=SKLearnMaterializer)
def train_model(vectors):
    #mlflow.sklearn.autolog()
    model_instance = ModelTrain()
    model = model_instance.train(vectors)

    wrapped_model = NearestNeighborsWrapper(model)

        #Ensure clean MLflow run
    if mlflow.active_run():
        mlflow.end_run()
    try:
        print('mlflow tracking uri', mlflow.get_tracking_uri())
        # # Set tracking URI explicitly (important for ZenML integration)
        # mlflow.set_tracking_uri("http://127.0.0.1:5000")
        # mlflow.set_experiment("movie_recommender_model")

        with mlflow.start_run() as run:

            Log parameters
            mlflow.log_params({
                'n_neighbors': 6,
                'metric': 'cosine'
            })
            # Log the wrapped model to MLflow
            model_info = mlflow.sklearn.log_model(
                sk_model=wrapped_model,
                artifact_path="movie_recommender",
                registered_model_name="movie_recommender_model",
                input_example=vectors[:1],
                signature=mlflow.models.infer_signature(
                    vectors[:1],
                    model.kneighbors(vectors[:1])[1])
            )



            print(f"Logged to MLflow run: {run.info.run_id}")
            print('Type of returned model:', type(model))  # Should be NearestNeighbors

        print(f"Model URI: {model_info.model_uri}")

    except Exception as e:
        print(f"Failed to log model: {str(e)}")

    return model
