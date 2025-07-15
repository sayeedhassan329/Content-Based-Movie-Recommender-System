from zenml import step
import mlflow

import logging


from model.evaluation import MovieEvaluation


@step(experiment_tracker="mlflow_tracker")
def evaluation(model) -> float:
    """
    Zenml step for evaluating movie recommendations.
    """
    try:
        logging.info("Starting evaluation")
        eval = MovieEvaluation()
        mean_div_score = eval.calculate_diversity_bow(model)

        print(f"\nAverage diversity across all samples: {mean_div_score:.3f}")
        print("(Lower values indicate more diverse recommendations)")

        mlflow.log_metric("diversity_score", mean_div_score)

        if not isinstance(mean_div_score, (int, float)):
            raise ValueError("Evaluation must return a numeric value")
        return float(mean_div_score)

    except Exception as e:
        logging.error(f"Error during zenml step: evaluation: {e}")
        raise e
