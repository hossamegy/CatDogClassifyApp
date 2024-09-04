import os
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def remove_corrupted_images(directory):
    num_removed = 0
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                img = Image.open(file_path)
                img.verify()
            except (IOError, SyntaxError) as e:
                print(f'Removing corrupted image: {file_path}')
                os.remove(file_path)
                num_removed += 1
    print(f"Total corrupted images removed: {num_removed}")


    
def generate_img_data(path):
    remove_corrupted_images("./dataset")
    img_gen = ImageDataGenerator(
       rescale=1./255,
       rotation_range = 15,
       horizontal_flip = True,
       zoom_range = 0.2,
       shear_range = 0.1,
       fill_mode = 'reflect',
       width_shift_range = 0.1,
       height_shift_range = 0.1,
       validation_split=0.2
    )

    train_data_gen = img_gen.flow_from_directory(
        path,
        target_size=(256, 256),
        subset="training",
        color_mode='rgb',
        shuffle=True,
        batch_size=8
    )
    validation_data_gen = img_gen.flow_from_directory(
        path,
        target_size=(256, 256),
        subset="validation",
        color_mode='rgb',
        shuffle=True,
        batch_size=8
    )
    return train_data_gen, validation_data_gen