# your_streamlit_app.py

import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

# Load the pre-trained model
model = tf.keras.models.load_model("models/imageclassifier.h5")

st.title("Image Mood Classifier")

uploaded_image = st.file_uploader("Upload an image of a person...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Read the uploaded image
    image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Resize the image to the model's input size (adjust as needed)
    image = cv2.resize(image, (256, 256))
  # Normalize the image

    # Make a prediction
    prediction = model.predict(np.expand_dims(image/255, 0))

    # Display the result
    st.subheader("Prediction:")
    if prediction > 0.5:
        st.write("The person in the image is predicted to be **Sad**.")
    else:
        st.write("The person in the image is predicted to be **Happy**.")

# Add some additional text to explain how to use the app
st.markdown(
    """
    ### Instructions:
    1. Upload an image of a person.
    2. Click the 'Classify' button to predict the person's mood.
    """
)
