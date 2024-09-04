from controllers.preprocessing import Preprocessor
from loader import loaded_model
import numpy as np
from ResponseSignels import ResponseSignal
from fastapi import UploadFile


class Inference:

    async def inference_func(self, file: UploadFile):
        name_classes = {0: 'Cat', 1: "Dog"}
        
        if not file.filename.lower().endswith(ResponseSignal().FILE_TYPE):
            return ResponseSignal().FILE_TYPE_NOT_SUPPORTED
       
        preprocessor = Preprocessor()
        preprocessed_image = await preprocessor.load_and_preprocess_image(file)
        
        # Make prediction
        predictions_probs = loaded_model.predict(preprocessed_image)
        predictions = np.argmax(predictions_probs)
        return name_classes[predictions]
