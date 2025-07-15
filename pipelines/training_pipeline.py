from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from zenml.pipelines import pipeline
from steps.clean_data import clean_data
from steps.ingest_data import ingest_data
from steps.model_train import train_model
from steps.evaluation import evaluation
from steps.data_transformation import transform_data
from materializers.custom_materializer import SKLearnMaterializer


docker_settings = DockerSettings(required_integration=[MLFLOW])

@pipeline(enable_cache=False, settings={'docker': docker_settings})
def train_pipeline(data_path1: str, data_path2: str):

    data = ingest_data(data_path1, data_path2)
    final_df = clean_data(data)
    vectors = transform_data(final_df)
    model = train_model(vectors)

    #score = evaluation(model)
    recommender = MovieRecommender(
            model_path="artifacts/trained_model/model.joblib",
            data_path='artifacts/cleaned_data/cleaned_data.joblib',
            vectors_path='artifacts/vectors/vectors.joblib'
        )

        # Get recommendations (for testing)
    recommendations = recommender.recommend_movies('Batman Begins')
    print(f"Recommendations:\n {recommendations}")
