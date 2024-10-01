import streamlit as st
import requests
from PIL import Image
import numpy as np

st.title("Brain MRI Metastasis Segmentation")

uploaded_file = st.file_uploader("Upload a Brain MRI Image", type=["tif"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Image.', use_column_width=True)
    
    files = {"file": uploaded_file.getvalue()}
    response = requests.post("http://localhost:8000/predict/", files=files)
    
    prediction = np.array(response.json()["prediction"])
    st.image(prediction, caption='Predicted Metastasis Mask', use_column_width=True)
