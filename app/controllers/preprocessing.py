from PIL import Image
import numpy as np
import io

class Preprocessor:

    async def load_and_preprocess_image(self, file):
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes))
        img = img.resize((256, 256))
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
