import os
import sys
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st


import os
import subprocess
from pathlib import Path

# Add this BEFORE your MovieRecommender class initialization
if not Path("artifacts/trained_model/model.joblib").exists():  # Update path
    subprocess.run(["dvc", "pull", "-r", "origin"], check=True)

class MovieRecommender:
    def __init__(self, model_path: str, data_path: str, vectors_path: str):
        """Initialize with paths to model and data"""

        self.model = joblib.load(model_path)
        self.data = joblib.load(data_path)
        self.vectors = joblib.load(vectors_path)



    def recommend_movies(self, movie_title):
        # Find movie index
        movie_index = self.data[self.data['title'] == movie_title].index[0]

        movie_vector = self.vectors[movie_index].reshape(1, -1)
        # Get nearest neighbors using KNN
        distances, indices = self.model.kneighbors(movie_vector)

        # Return top 5 similar movies (excluding itself)
        similar_movies = self.data.iloc[indices[0][1:6]]['title']
        return similar_movies.to_list()


    def recommendation_engine(self, similar_movies):

        st.text(similar_movies[0])
        st.text(similar_movies[1])
        st.text(similar_movies[2])
        st.text(similar_movies[3])
        st.text(similar_movies[4])


if __name__ == "__main__":
    st.header("Movie Recommender System")
    st.text('This is a content based movie recommender system.')
    # Initialize MovieRecommender
    recommender = MovieRecommender(
            model_path="artifacts/trained_model/model.joblib",
            data_path='artifacts/cleaned_data/cleaned_data.joblib',
            vectors_path='artifacts/vectors/vectors.joblib'
        )


    # Streamlit app
    # User input
    data = joblib.load('artifacts/cleaned_data/cleaned_data.joblib')
    movie_titles = data['title']

    selected_movie = st.selectbox('Select a movie', movie_titles)
    if st.button('Recommend'):

        similar_movies = recommender.recommend_movies(selected_movie)
        recommender.recommendation_engine(similar_movies)
