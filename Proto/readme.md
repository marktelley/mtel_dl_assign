# Check Before You Bin!


Check Before You Bin! is a web-based application that helps you determine whether an item is recyclable or not. By uploading an image of the item, the application uses machine (deep learning) learning models to classify the item and provide a recycling label. It combines the power of TensorFlow, Keras, Streamlit, and other libraries to create an intuitive and interactive user interface.

## Introduction

Recycling plays a crucial role in reducing waste and promoting sustainability. However, it's not always clear which items can be recycled and which cannot. Check Before You Bin! aims to solve this problem by leveraging deep learning models to classify items and provide recycling labels. By simply uploading an image, you can quickly determine whether an item should be recycled or not.

The application uses three models: TrashNet (based on the [Trashnet Dataset](https://github.com/garythung/trashnet)), Kaggle (Based on the [Kaggle Dataset](https://www.kaggle.com/techsash/waste-classification-data)), and ResNet (Pretrained model - InceptionResNetV2). TrashNet is trained to classify items into seven categories: plastic, cardboard, compost, glass, trash, paper, and metal. Kaggle is a binary classifier trained to identify whether an item is recyclable or not. ResNet is a pre-trained model that helps map the labels obtained from Kaggle to corresponding recycling labels. By combining the predictions from these models, the application provides a final recycling label for the item.

## Logic
The `determine_final_label` function takes three input labels: `kaggle_label`, `trashnet_label`, and `resnet_label`. It determines the final label based on the following logic:

- If the `kaggle_label` is "Not Recycling" and both `trashnet_label` and `resnet_label` are "Recycling", the `final_label` is set to "Recyclable".
- If the `kaggle_label` is "Recycling" and `resnet_label` is "Recycling", the `final_label` is set to "Recyclable".
- If the `kaggle_label` is "Recycling" and `trashnet_label` is "Recycling", the `final_label` is set to "Recyclable".
- In all other cases, the `final_label` is set to "Not Recyclable".

The determined `final_label` is then returned by the function.

## Prerequisites

Before running the application, make sure you have the following dependencies installed:

- Python 3.x
- Streamlit
- TensorFlow
- Keras
- PIL (Python Imaging Library)
- NumPy
- Pandas
- pyngrok

You can install the required dependencies by running the following command:

````{}
pip install streamlit tensorflow keras pillow numpy pandas pyngrok
````


## Usage

To run the application, follow these steps:

1. Clone the repository or download the source code files.

2. Open a terminal or command prompt and navigate to the directory where the source code files are located.

3. Run the following command to start the Streamlit application:

    ````{}
    streamlit run app.py
    ````

4. Wait for the application to start. It will display a URL that you can use to access the web interface.

5. Open a web browser and enter the URL provided by the application.

6. Upload an image file (in JPG, JPEG, or PNG format) using the file upload button on the web interface.

7. The application will process the image and display the uploaded image along with the classification result.

8. The classification result will indicate whether the item in the image is recyclable or not.


## Acknowledgements

This application uses pre-trained models and weights from TensorFlow and Keras. The `InceptionResNetV2` model is loaded from the `keras.applications.inception_resnet_v2` module, and the `model_trashnet` and `model_kaggle` are loaded using the `tf.keras.models.load_model()` function. The code also includes a CSV file (`imagenet_labels_updated.csv`) that is used for mapping imagenet labels.

## File Dependencies

This application relies on the following files:

- `app.py`: The main Python script that contains the code for the Streamlit application.

- `trashnet.h5`: The model weights file for the TrashNet model.

- `model_3.h5`: The model weights file for the Kaggle model.

- `imagenet_labels_updated.csv`: The CSV file that contains the label mapping data for the ResNet model.

- `key.txt`: A text file that contains the ngrok authentication token (Not provided in the Github Repo)

Make sure that these files are present in the appropriate locations specified in the code.



## Customisation

You can customise the behavior of the application by modifying the code in the `app.py` file. Here are some points to consider:

- You can change the page configuration settings by modifying the `st.set_page_config()` function call. This allows you to customise the title, icon, layout, and sidebar state of the web interface.

- The application loads pre-trained models and weights from specific file paths. If you have different model files, make sure to update the file paths in the `model_trashnet = tf.keras.models.load_model()` and `model_kaggle = tf.keras.models.load_model()` lines to point to your model files.

- Similarly, if you have a different CSV file containing label data, update the `csv_file` variable with the appropriate file path.

- The code includes several helper functions (`preprocess_image()`, `rename_model_layers()`, `map_resnet_label_to_marks()`, `map_marks_label_to_recycling_label()`, `determine_final_label()`) that handle preprocessing, label mapping, and final label determination. If you want to modify the behavior of these functions, you can do so in the corresponding code

    ````{}
    .
    ├── app.py
    ├── trashnet.h5
    ├── model_3.h5
    ├── imagenet_labels_updated.csv
    ├── key.txt
    ````

Please make sure to organise the files in the correct structure for the application to work properly.





