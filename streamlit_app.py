import streamlit as st
import numpy as np
import pandas as pd
import tensorflow
import tensorflow as tf
import shutil
import keras
import requests 
import io
import cv2


# Import your pre-trained model and libraries

# download model from Dropbox, cache it and load the model into the app 
@st.cache(allow_output_mutation=True)
def download_model(url):
  """Downloads a zipped model file from the specified URL using requests."""
  model_response = requests.get(url)
  model_response.raise_for_status()  # Raise error for failed downloads
  return model_response.content, model_response


# Function to load and preprocess image
def load_image(image_file):
    # Load image and apply preprocessing steps
    img = cv2.resize(cv2.imread(image_file), (224,224))
    return img

# Function to predict pneumonia using the model
def predict_pneumonia(image):
    # We use our model to predict pneumonia and get probability
    test_image = np.asarray(image)
    test_image = np.expand_dims(test_image, axis = 0)
    prediction = loaded_model.predict(test_image)
    if(prediction[0] > 0.5):
        statistic = prediction[0] * 100 
        #print("This image is %.3f percent %s"% (statistic, "P N E U M O N I A"))
        return "P N E U M O N I A" , statistic
    else:
        statistic = (1.0 - prediction[0]) * 100
        #print("This image is %.3f percent %s" % (statistic, "N O R M A L"))
        return "N O R M A L" , statistic

model_url = "https://www.dropbox.com/scl/fi/9aazpmx6wnahturotqmk6/my_pneumonia_detection_model.h5?rlkey=lb51utq5dxgozq89hs0s202ne&st=xrslo3jf&dl=1"
model_bytes, content = download_model(model_url)
loaded_model = tensorflow.keras.models.load_model(model_bytes)

# Main app
def main():
    # Create tabs
    tab1, tab2 = st.tabs(["Overview", "Prediction"])

    # Overview tab content
    with tab1:
        st.title("Pneumonia Image Prediction App")
        st.write("This app helps identify pneumonia in chest X-ray images. Open the main tab, upload a 'jpg' or 'png' image and click 'predict' ")

        # Overview section
        st.header("What is Pneumonia?")
        st.write("Pneumonia is an infection that inflames the air sacs in one or both lungs. These air sacs, called alveoli, fill with fluid or pus (a thick, yellowish-white liquid), making it difficult to breathe.")
        st.write("Symptoms:")
        st.write("Shortness of breath that worsens with activity or even at rest")
        st.write("Diagnosis:")
        st.write("Medical history and physical exam, Chest X-ray, Sputum test, Pulse oximetry")
        st.write("")    
        st.image("pneumo-true.jpeg", caption="X-ray image of an unhealthy lung")
        st.image("pneumo-false.jpeg", caption="X-ray image of a healthy lung")

    # Prediction tab content
    with tab2:
        st.write("Upload an Image to test for pneumonia and click.")
        st.write("Accepted file formats: jpg, png" )
        uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "png"])
        if uploaded_file is not None:
            image = load_image(uploaded_file)
            st.image(image, caption="Uploaded Image")

            # Prediction section
            if st.button("Predict Pneumonia"):
                predictions, probs = predict_pneumonia(image)
                st.subheader("Prediction Results:")
                #st.image(image, caption="Original Image")

                # Display segmented image (optional, depends on our model)
                if "segmented_image" in predictions:
                    st.image(predictions["segmented_image"], caption="Segmented Image")

                # Display predicted category and probability
                st.write(f"Predicted Category: {predictions}")
                st.write(f"Probability: {probs:.2f}")

if __name__ == "__main__":
    main()
