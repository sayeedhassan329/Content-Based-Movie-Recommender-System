from pipelines.training_pipeline import train_pipeline

from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri



if __name__ == "__main__":
    pipeline = train_pipeline(data_path1="data/tmdb_5000_movies.csv", data_path2='data/tmdb_5000_credits.csv')




    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs.")
