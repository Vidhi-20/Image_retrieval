# app/app.py

import sys
import os

# Set the path to the CBIR_Project root and the src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from retrieval import retrieve_similar_images

import streamlit as st
from PIL import Image
import torch
from torchvision import transforms  # Add this import for transforms
from retrieval import retrieve_similar_images
from dbn import DeepBeliefNetwork
from data_loader import load_data

# Load trained DBN model and features
dbn_model = DeepBeliefNetwork([1024, 512, 256])  # Adjust layer sizes as needed
_, test_loader = load_data()

# Streamlit App
st.title("Image Retrieval System")
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    query_image = Image.open(uploaded_file).convert("L")
    st.image(query_image, caption="Query Image")
    
    # Convert the query image to a tensor
    query_image = transforms.ToTensor()(query_image).unsqueeze(0)  # Now this line should work

    # Retrieve similar images
    top_images = retrieve_similar_images(query_image, dbn_model, test_loader)
    st.write("Top similar images:")
    for img in top_images:
        st.image(img, caption="Similar Image")
