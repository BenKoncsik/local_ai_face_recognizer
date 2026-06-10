"""Deep-learning face recognition engine.

This package implements the "new" recognition path: a neural-network
classifier head trained on top of the deep CNN face embeddings, retrained
from the user's labeled faces on every run (continual learning).

Modules
-------
dataset     – builds the training dataset from labeled faces in the DB
classifier  – DeepFaceClassifier: MLP ensemble + open-set rejection
"""

from app.deep.classifier import DeepFaceClassifier, DeepPrediction, DeepTrainingReport
from app.deep.dataset import TrainingDataset, build_training_dataset

__all__ = [
    "DeepFaceClassifier",
    "DeepPrediction",
    "DeepTrainingReport",
    "TrainingDataset",
    "build_training_dataset",
]
