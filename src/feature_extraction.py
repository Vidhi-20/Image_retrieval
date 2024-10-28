# src/feature_extraction.py
import torch
from dbn import DeepBeliefNetwork

def extract_features(dbn_model, images):
    features = []
    for img in images:
        feature = dbn_model(img.unsqueeze(0)).detach().numpy()  # Assuming batch size 1
        features.append(feature)
    return features
