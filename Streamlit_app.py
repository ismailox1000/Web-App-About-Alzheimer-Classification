# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 08:15:45 2023
ISMAIL OUBAH 

@author: is-os
"""
#now the fun part which is build a web app to our model 

import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('C:\\Users\\is-os\\ADv2.h5')

# Load the test data directory and get class names
test_data = tf.keras.preprocessing.image_dataset_from_directory(
    'C:\\Users\\is-os\\output\\test', image_size=(128, 128), seed=159, batch_size=64)
class_names = test_data.class_names

# Function to save uploaded image and return file path
def save_uploaded_image(uploaded_file):
    file_path = f"C:\\Users\\is-os\\Desktop\\Dataset\\{uploaded_file.name}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as img_file:
        img_file.write(uploaded_file.getbuffer())
    return file_path

# Function to predict image class and confidence
def predict_image_class(file_path):
    test_image = tf.keras.preprocessing.image.load_img(
        file_path, target_size=(128, 128))
    array_image = tf.keras.preprocessing.image.img_to_array(test_image)
    array_image = tf.keras.backend.expand_dims(array_image, axis=0)
    prediction = model.predict(array_image)
    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = round(100 * (np.max(prediction[0])), 2)
    return predicted_class, confidence

# Streamlit app code
st.title("Alzheimer's Disease Classifier")

uploaded_file = st.file_uploader('Choose an image.', type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    file_path = save_uploaded_image(uploaded_file)
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded image')
    if st.button('See prediction'):
        st.spinner('Please wait a moment...')
        predicted_class, confidence = predict_image_class(file_path)
        st.write(f"Your disease stage is {predicted_class} and the confidence is {confidence}%")
