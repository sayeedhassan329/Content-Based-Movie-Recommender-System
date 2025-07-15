from zenml import pipeline, step
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from typing import Optional, cast, Dict, Any
from steps.clean_data import clean_data
from steps.ingest_data import ingest_data
from steps.model_train import train_model

from steps.data_transformation import transform_data

from steps.model_train import train_model
import pandas as pd
import numpy as np
import mlflow
import logging


from zenml.integrations.constants import MLFLOW
from zenml.config import DockerSettings
docker_settings = DockerSettings(required_integrations=[MLFLOW])


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    data_path1: str = "data/tmdb_5000_movies.csv",
    data_path2: str = "data/tmdb_5000_credits.csv",
    workers: int = 3,
    timeout: int = 300
):
    """Continuous deployment pipeline for movie recommendation system.

    Args:
        data_path1: Path to movies dataset
        data_path2: Path to credits dataset
        min_accuracy: Minimum accuracy threshold for deployment
        workers: Number of workers for deployment
        timeout: Timeout for deployment
    """
    # Data processing steps
    data = ingest_data(data_path1, data_path2)
    final_df = clean_data(data)
    vectors = transform_data(final_df)
    model = train_model(vectors)


    #Model deployment with MLflow
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=True,
        workers=workers,
        timeout=timeout,


    )


@step
def get_recommendations(
    movie_title: str,
    dataset_path: str = "data/tmdb_5000_movies.csv"
) -> Dict[str, Any]:
    """Get movie recommendations from deployed model.

    Args:
        movie_title: Title of movie to get recommendations for
        dataset_path: Path to movie dataset

    Returns:
        Tuple of (input_movie_title, list_of_recommendations)
    """
    # Load the dataset to get movie indices
    df = pd.read_csv(dataset_path)

    # Get the MLflow deployment service
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()
    existing_services = model_deployer.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        model_name="movie_recommender_model",
        running=False
    )

    if not existing_services:
        raise RuntimeError(
            "No MLflow prediction service deployed"
        )
    print('Existing_services: ', existing_services)
    print('Type of existing services: ', type(existing_services))
    service = existing_services[0]

        # Ensure the service is running
    if not service.is_running:
        logging.info(f"Starting MLflow service {service.uuid}...")
        service.start(timeout=120)

        if not service.is_running:
            raise RuntimeError(
                f"Failed to start MLflow prediction service {service.uuid}. "
                f"Current state: {service.status.state.value}"
            )

    # if not services:
    #     raise RuntimeError("No MLflow deployment service found!")

    # service = services[0]
    # if not service.is_running:
    #     raise RuntimeError("MLflow service is not running!")

    # Prepare input data (you'll need to adapt this based on your model's requirements)
    movie_idx = df[df['title'] == movie_title].index[0]
    input_data = np.array([movie_idx]).reshape(1, -1)

    # Make prediction
    response = service.predict(input_data)
    distances, indices = response[0], response[1]

    # Get recommended movie titles
    recommendations = [df.iloc[idx]['title'] for idx in indices[0]]

    #return (movie_title, recommendations)
    # Convert to JSON-serializable format
    return {
        "input_movie": movie_title,
        "recommendations": recommendations  # This should already be a list of strings
    }


@step
def display_recommendations(recommendations: Dict[str, Any]):
    """Display recommendations to the user."""
    input_movie = recommendations["input_movie"]
    recommended_movies = recommendations["recommendations"]

    print(f"\n[bold]Recommendations for '{input_movie}':[/bold]")
    for i, movie in enumerate(recommended_movies, 1):
        print(f"{i}. {movie}")

@pipeline(enable_cache=False)
def inference_pipeline(
    pipeline_name: str,
    pipeline_step_name: str,
    movie_title: str,
    dataset_path: str = "data/tmdb_5000_movies.csv"
):
    """Inference pipeline for getting movie recommendations.

    Args:
        pipeline_name: Name of the deployment pipeline
        pipeline_step_name: Name of the deployment step
        movie_title: Title of movie to get recommendations for
        dataset_path: Path to movie dataset
    """
    recommendations = get_recommendations(movie_title, dataset_path)
    display_recommendations(recommendations)
