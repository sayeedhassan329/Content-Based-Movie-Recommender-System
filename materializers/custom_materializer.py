# from zenml.materializers.base_materializer import BaseMaterializer
# import joblib
# import os

# from sklearn.neighbors import NearestNeighbors

# class SKLearnMaterializer(BaseMaterializer):
#     ASSOCIATED_TYPES = (NearestNeighbors,)

#     def load(self, data_type):
#         """Load a sklearn model from disk"""
#         return joblib.load(os.path.join(self.uri, "model.joblib"))

#     def save(self, model):
#         """Save a sklearn model to disk"""
#         joblib.dump(model, os.path.join(self.uri, "model.joblib"))


# materializers/sklearn_materializer.py
# from zenml.materializers.base_materializer import BaseMaterializer
# from sklearn.neighbors import NearestNeighbors
# import joblib
# import os

# class SKLearnMaterializer(BaseMaterializer):
#     ASSOCIATED_TYPES = (NearestNeighbors,)

#     def load(self, data_type) -> NearestNeighbors:
#         """Load from artifact store"""
#         return joblib.load(os.path.join(self.uri, "model.joblib"))

#     def save(self, model: NearestNeighbors):
#         """Save to artifact store"""
#         os.makedirs(self.uri, exist_ok=True)
#         joblib.dump(model, os.path.join(self.uri, "model.joblib"))


# from zenml.materializers.base_materializer import BaseMaterializer
# from sklearn.neighbors import NearestNeighbors
# import joblib
# import os
# from typing import Type

# class NearestNeighborsMaterializer(BaseMaterializer):
#     ASSOCIATED_TYPES = (NearestNeighbors,)
#     ASSOCIATED_ARTIFACT_TYPE = "model"

#     def load(self, data_type: Type[NearestNeighbors]) -> NearestNeighbors:
#         """Load the materialized NearestNeighbors model."""
#         return joblib.load(os.path.join(self.uri, "model.joblib"))

#     def save(self, model: NearestNeighbors) -> None:
#         """Save the NearestNeighbors model."""
#         joblib.dump(model, os.path.join(self.uri, "model.joblib"))



# from zenml.materializers.base_materializer import BaseMaterializer
# from sklearn.neighbors import NearestNeighbors
# import joblib
# import os

# class SklearnMaterializer(BaseMaterializer):
#     ASSOCIATED_TYPES = (NearestNeighbors,)
#     ASSOCIATED_ARTIFACT_TYPE = "model"

#     def load(self, data_type) -> NearestNeighbors:
#         """Load from artifact store"""
#         return joblib.load(os.path.join(self.uri, "model.joblib"))

#     def save(self, model: NearestNeighbors):
#         """Save to artifact store"""
#         os.makedirs(self.uri, exist_ok=True)
#         joblib.dump(model, os.path.join(self.uri, "model.joblib"))


from zenml.materializers.base_materializer import BaseMaterializer
from sklearn.neighbors import NearestNeighbors
import joblib
import os
from zenml.enums import ArtifactType

class SKLearnMaterializer(BaseMaterializer):
    ASSOCIATED_TYPES = (NearestNeighbors,)
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.MODEL  # Use enum value instead of string

    def load(self, data_type) -> NearestNeighbors:
        """Load from artifact store"""
        return joblib.load(os.path.join(self.uri, "model.joblib"))

    def save(self, model: NearestNeighbors):
        """Save to artifact store"""
        os.makedirs(self.uri, exist_ok=True)
        joblib.dump(model, os.path.join(self.uri, "model.joblib"))
