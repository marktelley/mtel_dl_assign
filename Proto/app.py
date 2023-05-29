import streamlit as st
import tensorflow as tf
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input, decode_predictions
from PIL import Image
import numpy as np
import pandas as pd
from pyngrok import ngrok
import subprocess
import psutil

print(tf.__version__)
# Set page configuration
st.set_page_config(
    page_title="Check before you bin!",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="auto"
)

# Apply custom CSS style for font
st.markdown(
    """
    <style>
    h1, h2, h3, h4 {
        font-family: monospace !important;
    }
    body {
        font-family: monospace !important;
        background-color: rgb(255, 248, 230) !important;
    }
     [role="alert"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Function to preprocess the image
def preprocess_image(image, target_size):
    img = image.resize(target_size)
    img = np.array(img) / 255.0  # Normalize the image
    return img

# Renames the layers of a model with a given prefix
def rename_model_layers(model, model_name):
    for layer in model.layers:
        layer._name = model_name + "_" + layer._name
    return model

# Maps the ResNet label to the corresponding Marks label
def map_resnet_label_to_marks(resnet_label, csv_file):
    # Load the label data from the CSV file
    label_data = pd.read_csv(csv_file)

    # Find the matching row where 'resnet_label' matches the 'Label' column
    matching_row = label_data[label_data['Label'] == resnet_label]

    # Get the corresponding 'Marks_Label' value from the matching row
    marks_label = matching_row['Marks_Label'].values[0] if not matching_row.empty else "Unknown"

    return marks_label

# Maps the Marks label to the corresponding recycling label
def map_marks_label_to_recycling_label(marks_label):
    recycling_mapping = {
        'trash': 'Not Recyclable',
        'glass': 'Recycling',
        'metal': 'Recycling',
        'cardboard': 'Recycling',
        'paper': 'Recycling',
        'plastic': 'Recycling',
        'compost': 'Not Recyclable'
    }
    recycling_label = recycling_mapping.get(marks_label, 'Unknown')

    return recycling_label

# Determines the final label based on the predictions from different models
def determine_final_label(kaggle_label, trashnet_label, resnet_label):
    # Check if Kaggle label is "Not Recycling" and both TrashNet and ResNet are "Recycling"
    if kaggle_label == "Not Recycling" and trashnet_label == "Recycling" and resnet_label == "Recycling":
        final_label = "Recyclable"
    elif kaggle_label == "Recycling" and resnet_label == "Recycling":
        final_label = "Recyclable"
    elif kaggle_label == "Recycling" and trashnet_label == "Recycling":
        final_label = "Recyclable"
    else:
        final_label = "Not Recyclable"

    return final_label

# Load the models and weights
model_trashnet = tf.keras.models.load_model('/Users/marktelley/PycharmProjects/mproto/trashnet.h5')
model_trashnet = rename_model_layers(model_trashnet, "trashnet")

model_kaggle = tf.keras.models.load_model('/Users/marktelley/PycharmProjects/mproto/model_3.h5')
model_kaggle = rename_model_layers(model_kaggle, "kaggle")

model_imagenet = InceptionResNetV2(weights='imagenet')
model_imagenet = rename_model_layers(model_imagenet, "resnet")

csv_file = '/Users/marktelley/PycharmProjects/mproto/imagenet_labels_updated.csv'

# Streamlit code
st.title('Check before you bin!')
st.subheader('Check if the item is recyclable or not.')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    processed_image_trashnet = preprocess_image(image, (300, 300))
    processed_image_inception = preprocess_image(image, (299, 299))
    processed_image_kaggle = preprocess_image(image, (180, 180))

    # Make predictions using the models
    preds_imagenet = model_imagenet.predict(np.array([processed_image_inception]))
    decoded_preds_imagenet = decode_predictions(preds_imagenet, top=1)

    preds_trashnet = model_trashnet.predict(np.array([processed_image_trashnet]))
    preds_kaggle = model_kaggle.predict(np.array([processed_image_kaggle]))

    # ResNet
    resnet_label = decoded_preds_imagenet[0][0][1]
    marks_label = map_resnet_label_to_marks(resnet_label, csv_file)
    recycling_label_resnet = map_marks_label_to_recycling_label(marks_label)

    # TrashNet
    trashnet_label = ['plastic', 'cardboard', 'compost', 'glass', 'trash', 'paper', 'metal'][np.argmax(preds_trashnet)]
    recycling_label_trashnet = map_marks_label_to_recycling_label(trashnet_label)

    # Kaggle
    kaggle_label = {0: "Not Recycling", 1: "Recycling"}.get(np.argmax(preds_kaggle), "Unknown Class")

    # FINAL LABEL
    final_label = determine_final_label(kaggle_label, recycling_label_trashnet, recycling_label_resnet)

    # SHOW TO USER
    if marks_label != "trash":
        st.write("\nThis looks like", marks_label, "Put that in the compost" if marks_label == 'compost' else "")
    st.write("\nThis item is", final_label)

# Find and terminate ngrok process
for proc in psutil.process_iter(['name', 'cmdline']):
    if proc.info['name'] == 'ngrok' and 'http' in proc.info['cmdline']:
        proc.terminate()

# Start ngrok
subprocess.Popen(['ngrok', 'http', '8502'])

# Run ngrok diagnose command
result = subprocess.run(['ngrok', 'diagnose'], capture_output=True, text=True)

# Get the output of the command
output = result.stdout

# Print the output
print(output)

# Read the contents of the text file
with open("/Users/marktelley/PycharmProjects/mproto/key.txt", "r") as file:
    auth_token = file.read().strip()

# Set ngrok auth token`
ngrok.set_auth_token(auth_token)
# Start ngrok tunnel
public_url = ngrok.connect(addr="8501")
print(public_url)
