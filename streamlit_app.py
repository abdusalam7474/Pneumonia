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
import tempfile
from PIL import Image

# Import your pre-trained model and libraries

# download model from Dropbox, cache it and load the model into the app 
@st.cache(allow_output_mutation=True)
def download_model(url):
  """Downloads a zipped model file from the specified URL using requests."""
  model_response = requests.get(url)
  model_response.raise_for_status()  # Raise error for failed downloads
  model_data = io.BytesIO(model_response.content)
  with tempfile.NamedTemporaryFile(delete=False) as temp_file:
    temp_file.write(model_data.getbuffer())
    #model = keras.models.load_model(temp_file.name)
  
  return temp_file.name, model_response
  
def check_model(content):
  try:
    model_response = content
    #model_response.raise_for_status()  # Raise error for failed downloads

    # Extract file extension (assuming content-type header is present)
    content_type = model_response.headers.get('Content-Type')
    if content_type:
      file_extension = content_type.split('/')[-1]
    else:
      file_extension = "none available"
    st.write("All models downloaded successfully")
    return True, file_extension
  except requests.exceptions.RequestException as e:
    st.write(f"Download failed: {e}")
    return False, ""
    
# Function to load and preprocess image
def load_image(image_file):
    # Load image and apply preprocessing steps
    #img = cv2.resize(np.asarray(image_file), (224,224))
    img = Image.open(image_file)
    img = img.resize((224, 224))
    return img

# Function to predict pneumonia using the model
def predict_pneumonia(image, sel_model):
    # We use our model to predict pneumonia and get probability
    test_image = np.asarray(image)
    test_image = np.expand_dims(test_image, axis = 0)
    prediction = sel_model.predict(test_image)
    if(prediction[0] > 0.5):
        statistic = prediction[0] * 100 
        #print("This image is %.3f percent %s"% (statistic, "P N E U M O N I A"))
        return "P N E U M O N I A" , prediction #statistic[0]
    else:
        statistic = (1.0 - prediction[0]) * 100
        #print("This image is %.3f percent %s" % (statistic, "N O R M A L"))
        return "N O R M A L" , prediction #statistic[0]

cnn_url = "https://www.dropbox.com/scl/fi/9aazpmx6wnahturotqmk6/my_pneumonia_detection_model.h5?rlkey=lb51utq5dxgozq89hs0s202ne&st=xrslo3jf&dl=1"
vg_url = "https://www.dropbox.com/scl/fi/7gpyh72rgic9ecu6jjbxe/my_pneumonia_detection_model_mn.h5?rlkey=aiw6my4qkr5k0iz8jjqlcyene&st=jwu2ajx7&dl=1"
mobnet_url = "https://www.dropbox.com/scl/fi/8ehf3u74vcrrfydk5rktp/my_pneumonia_detection_model_cnn.h5?rlkey=lab5fq8unvo9oyg48tknil9y5&st=w3nonxxh&dl=1"

cnn_path, content = download_model(cnn_url)
vg_path, vg_content = download_model(cnn_url)
mn_path, mn_content = download_model(cnn_url)

check_model(content)
cnn_model = tensorflow.keras.models.load_model(cnn_path)
vg_model = tensorflow.keras.models.load_model(vg_path)
mn_model = tensorflow.keras.models.load_model(mn_path)

# Main app
def main():
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Overview", "Models", "Prediction"])

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
        st.write("Select Your choice of model")
        model_sel = st.selectbox("AI models", ["Custom CNN", "VG16", "MobileNet"])
        if model_sel == "Custom CNN":
           sel_model = cnn_model
        elif model_sel == "VG16":
           sel_model = vg_model
        elif model_sel == "MobileNet":
           sel_model = mn_model
    
  # Prediction tab content
    with tab3:
        st.write("Upload an Image to test for pneumonia and click.")
        st.write("Accepted file formats: jpg, png" )
        uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "png"])
        if uploaded_file is not None:
            image = load_image(uploaded_file)
            st.image(image, caption="Uploaded Image")

            # Prediction section
            if st.button("Predict Pneumonia"):
                predictions, probs = predict_pneumonia(image, sel_model)
                st.subheader("Prediction Results:")
                #st.image(image, caption="Original Image")

                # Display segmented image (optional, depends on our model)
                if "segmented_image" in predictions:
                    st.image(predictions["segmented_image"], caption="Segmented Image")

                # Display predicted category and probability
                st.write(f"Predicted Category: {predictions}")
                st.write(f"Probability: {probs}")

if __name__ == "__main__":
    main()
