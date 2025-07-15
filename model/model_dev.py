

from sklearn.neighbors import NearestNeighbors
import joblib
import os
import logging
import mlflow
import numpy as np
from mlflow.pyfunc import PythonModel
import mlflow.sklearn





class ModelTrain:
    def train(self, vectors):

            model = NearestNeighbors(n_neighbors=6, metric='cosine')
            model.fit(vectors)

            # Create wrapper for MLflow


            # Save the raw sklearn model (not the wrapper)
            root_dir = os.getcwd()
            dir_path = os.path.join(root_dir, 'artifacts', 'trained_model')
            os.makedirs(dir_path, exist_ok=True)
            filename = os.path.join(dir_path, 'model.joblib')
            joblib.dump(model, filename)  # Save the sklearn model, not wrapper
            logging.info(f"Model saved to {filename}")


            return model# Return the sklearn model, not the wrapper
