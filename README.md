# **Cat vs Dog Image Classification System**
## **Project Overview**
The Cat vs Dog Image Classification System is a machine learning application designed to classify images as either 'Cat' or 'Dog'. The system integrates data preprocessing, custom model training, and real-time prediction via a web API.

## **Features**
* Data Preprocessing: Cleans and augments image data to ensure high-quality input for model training.
* Custom CNN Model: Utilizes a specialized convolutional neural network to accurately differentiate between cats and dogs.
* Training Optimization: Employs advanced training techniques to enhance model performance.
* Real-Time Inference: Provides an API for on-demand image classification.

## **Project Structure**
* ### **Data Preprocessing (img_generator.py)**   

    **Functionality:** Prepares the image dataset by removing corrupted images and applying augmentation techniques such as rotation, flipping, and zooming. This ensures the model is trained on high-quality, varied data.

* ### **Model Building (buildModel.py)** 
    
    **Functionality:** Defines and constructs a convolutional neural network tailored for binary classification tasks. The model architecture includes multiple convolutional layers followed by fully connected layers for final classification.
* ### **Model Training (train_model.py)** 
    
    **Functionality:** Manages the training process of the custom model, including data loading, model compilation, and training with validation. It also saves the trained model and records training history for future reference.
* ### **Inference (inference.py)**
    **Functionality:** Loads the trained model and performs predictions on new images. This script preprocesses input images and uses the model to classify them as 'Cat' or 'Dog'.
* ### **Deployment (app/ directory)**
 
    **Purpose:** Sets up a FastAPI-based web service to deploy the trained model for real-time inference.
    * controllers/preprocessing.py: Handles image preprocessing for inference.
    * controllers/inference.py: Manages the classification requests and returns results.
    * routers/base_router.py: Provides a basic endpoint to check server status.
    * routers/inference_router.py: Defines the endpoint for image classification.
    * app.py: Configures and starts the FastAPI  application.
* ### **6. Dependencies (requirements.txt) Purpose:** 
    
    Lists the necessary Python libraries required to run the project, including TensorFlow, Keras, Pillow, FastAPI, and Uvicorn.

## **Usage**
### **Training the Model**
python version 3.9
```bash
pip install -r requirements.txt
```
**1. Prepare Dataset:**

* Ensure your dataset is organized in the dataset directory, with subdirectories for training and validation images of cats and dogs.
to download full dataset: https://www.microsoft.com/en-us/download/details.aspx?id=54765

**2. train_model.py**

## **Performing Inference**
**1. python inference.py**

**2. Deploy with FastAPI:**
ensure you are in app dir
```bash
uvicorn app.main:app --reload
```
Access the API at http://localhost:8000/docs:

* **GET /:** Check server status.
* **POST /inference:** Upload an image file to receive a classification result (Cat or Dog).

## **API Documentation**
**GET /:** Confirms that the server is running.
**POST /inference:** Upload an image to classify. The response will include the predicted class label.

# **License**
This project is licensed under the Apache License 2.0. See the LICENSE file for details.

# **Acknowledgements**
* TensorFlow and Keras: For providing powerful tools for building and training the neural network.
* FastAPI: For its robust framework for creating and deploying the API.
* Pillow: For handling image processing tasks.