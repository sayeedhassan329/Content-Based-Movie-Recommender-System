from sklearn.feature_extraction.text import CountVectorizer
import os
import joblib
import logging

class DataTransformation:
    # def __init__(self):
    #     self.vectorizer = CountVectorizer(max_features=10000, stop_words='english')

    def transform(self, data):

        cv = CountVectorizer(max_features=5000, stop_words='english')
        print('transformed_new_df: ', data.shape)
        vectors = cv.fit_transform(data['tags']).toarray()
        print('vectors shape: ',vectors.shape)
        root_dir = os.getcwd()
        dir_path = os.path.join(root_dir, 'artifacts', 'vectors')
        os.makedirs(dir_path, exist_ok=True)
        filename = os.path.join(dir_path,'vectors.joblib')

        joblib.dump(vectors, filename)
        logging.info(f"Vectors saved to {filename}")

        return vectors
