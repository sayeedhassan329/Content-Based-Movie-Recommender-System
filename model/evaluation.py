
from joblib import load
from abc import ABC, abstractmethod

import numpy as np



class Evaluation(ABC):
    def __init__(self):
        """
        Initialize the evaluation with a model and data.
        """

    @abstractmethod
    def calculate_diversity_bow(self, model):
        """
        Calculate recommendation diversity using Bag of Words vectors"""
        pass



class MovieEvaluation(Evaluation):
    """
    Concrete class to evaluate movie recommendations.
    """


    def calculate_diversity_bow(self, model) -> float:



        vectors = load('artifacts/vectors/vectors.joblib')
        df = load('artifacts/cleaned_data/cleaned_data.joblib')
        #model = load('artifacts/trained_model/model.joblib')

        if len(df) > len(vectors):
            df = df.iloc[:len(vectors)]
        mean_similarities = []

        for index in df.index:
            if index >= len(vectors):
                continue
            movie_vector = vectors[index].reshape(1, -1)

            distances, _ = model.kneighbors(movie_vector)

        mean_similarities.append(distances.mean())


        diversity = np.mean(mean_similarities)
        print('mean diversity:', diversity)
        return diversity
