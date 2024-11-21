import streamlit as st
from PIL import Image
import numpy as np
from roboflow import Roboflow

def predict_image(image_path):
    rf = Roboflow(api_key="Qxa8egVXplQGNbVaaq5V")
    project = rf.workspace().project("chicken-pox")
    model = project.version(1).model

    try:
        # Use the model to make a prediction
        response = model.predict(image_path, confidence=40, overlap=30).json()

        # Parse the predictions
        predictions = response.get("predictions", [])

        # Check if chickenpox is detected
        if predictions:
            return ("Chickenpox detected! in the image")
        else:
            return ("No chickenpox detected in the image.")
    except Exception as e:
        print("Error occurred during prediction:", e)
    
# Streamlit app starts here
st.set_page_config(page_title="Image Classifier", layout="wide", initial_sidebar_state="collapsed")
st.title("Chickenpox Image Classifier")
st.markdown("Upload an image to classify it as **Chickenpox** or **Non-Chickenpox**.")

# Upload the image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "bmp", "tiff"])

if uploaded_file is not None:
    # Display the uploaded image with a fixed width
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=400)  # Set width to 400 pixels

    # Save the uploaded file temporarily
    image_path = "temp_image.jpg"
    image.save(image_path)
    

    # Predict the uploaded image
    with st.spinner("Classifying..."):
        prediction = predict_image( image_path)

    # Show the prediction result
    st.subheader(f"Prediction: {prediction}")
    
    # st.image(result_image, caption="Result", width=400)  # Set width to 400 pixels
    if(prediction == "Chickenpox detected! in the image"):
        st.error(prediction)
    else:
        st.success(prediction)

# Add an About Section
st.sidebar.title("About")
st.sidebar.info(
    """
    This app classifies images as Chickenpox or Non-Chickenpox using a Convolutional Neural Network.
    """
)
