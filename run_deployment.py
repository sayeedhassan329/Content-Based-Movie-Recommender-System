
from typing import cast
import click
from rich import print
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from pipelines.deployment_pipeline import continuous_deployment_pipeline, inference_pipeline


# Constants for CLI options
DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"

@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default=DEPLOY_AND_PREDICT,
    help="Option to run deployment, prediction, or both",
)
@click.option(
    "--movies-path",
    default="data/tmdb_5000_movies.csv",
    help="Path to movies dataset",
)
@click.option(
    "--credits-path",
    default="data/tmdb_5000_credits.csv",
    help="Path to credits dataset",
)
@click.option(
    "--movie-title",
    default="Batman Begins",
    help="Title of the movie to predict recommendations for",
)
def main(
    config: str,
    movies_path: str,
    credits_path: str,
    movie_title: str
):
    """Main function to run deployment and/or inference pipelines."""

    # Initialize MLflow model deployer
    mlflow_model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # Determine which pipelines to run
    deploy = config in (DEPLOY, DEPLOY_AND_PREDICT)
    predict = config in (PREDICT, DEPLOY_AND_PREDICT)

    # Run deployment pipeline if requested
    if deploy:
        print("[bold green]Running deployment pipeline...[/bold green]")
        continuous_deployment_pipeline(
            data_path1=movies_path,
            data_path2=credits_path,
        )

    # Run inference pipeline if requested
    if predict:
        print("[bold green]Running inference pipeline...[/bold green]")
        inference_pipeline(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
            movie_title=movie_title
        )

    # Print MLflow tracking URI
    print(
        f"\nMLflow Tracking: mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
    )

    # Check existing services
    existing_services = mlflow_model_deployer.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        model_name="movie_recommender_model"
    )

    # Report service status
    if existing_services:
        service = cast(MLFlowDeploymentService, existing_services[0])
        if service.is_running:
            print(
                f"[bold green]Model server is running at: {service.prediction_url}[/bold green]"
            )
            print(
                f"[bold green]To make predictions, use: python run_deployment.py --config predict --movie-title 'YOUR_MOVIE'[/bold green]"
            )
        elif service.is_failed:
            print(
                f"[bold red]Model server deployment failed: {service.status.last_error}[/bold red]"
            )
    elif predict:
        print(
            "[bold yellow]No active model server found! "
            "Run deployment first with: python run_deployment.py --config deploy[/bold yellow]"
        )

if __name__ == "__main__":
    main()
