from trained_model import Conv2Block, Classifier, MyModel
from tensorflow.keras.models import load_model

# Path to the saved model
model_path = r'trained_model\model'

# Load the model with custom objects
loaded_model = load_model(model_path, custom_objects={
    'Conv2Block': Conv2Block,
    'Classifier': Classifier,
    'MyModel': MyModel
})