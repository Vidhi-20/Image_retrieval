# src/retrieval.py
from data_loader import load_data
from feature_extraction import extract_features
from similarity import calculate_similarity
from dbn import DeepBeliefNetwork
import torch

def retrieve_similar_images(query_image, dbn_model, dataset_features):
    query_feature = extract_features(dbn_model, [query_image])[0]
    top_indices = calculate_similarity(query_feature, dataset_features)
    return top_indices
