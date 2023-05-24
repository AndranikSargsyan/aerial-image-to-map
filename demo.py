import numpy as np
import pandas as pd
import streamlit as st

st.title("Qartezator")

option = st.selectbox(
    'Choose model',
    ('Unet', 'Pix2pix','CycleGAN', 'DiscoGAN', 'LaMa'))


from PIL import Image

# Mock function 
def predict(image):
   
    return image



st.title("Aerial Image Prediction Demo")
st.write("Upload an aerial image and see the predicted map")

uploaded_file = st.file_uploader("Choose an aerial image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    
    predicted_map = predict(image)

    
    st.subheader("Predicted Map")
    st.image(predicted_map, caption="Predicted Map", use_column_width=True)


