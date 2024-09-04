from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from buildModel import Conv2Block, Classifier, MyModel

# Path to the saved model
model_path = 'trained_model/full_model_subclass3'

# Load the model with custom objects
loaded_model = load_model(model_path, custom_objects={
    'Conv2Block': Conv2Block,
    'Classifier': Classifier,
    'MyModel': MyModel
})

def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((256, 256))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def inference_func(image_path):
    name_classes = {0: 'Cat', 1: "Dog"}
    preprocessed_image = load_and_preprocess_image(image_path)
    predictions_probs = loaded_model.predict(preprocessed_image)
    predictions = np.argmax(predictions_probs)
    return name_classes[predictions]
