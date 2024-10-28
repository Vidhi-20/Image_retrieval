# src/similarity.py
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(query_feature, dataset_features):
    similarities = cosine_similarity([query_feature], dataset_features)
    return similarities[0].argsort()[-10:][::-1]  # Top 10 results
