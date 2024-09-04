import os 
from inference import inference_func

cats_image_folder = "testData\Cats"
dogs_image_folder = "testData\Dogs"


for filename in os.listdir(cats_image_folder):
    full_path = os.path.join(cats_image_folder, filename)
    predictions = inference_func(full_path)
    print(f"Predictions for {filename}: {predictions}")

for filename in os.listdir(dogs_image_folder):
    full_path = os.path.join(dogs_image_folder, filename)
    predictions = inference_func(full_path)
    print(f"Predictions for {filename}: {predictions}")
