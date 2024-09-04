import pickle
from buildModel import MyModel
from img_generator import generate_img_data
import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping

path = 'dataset'
training_data, validation_data = generate_img_data(path)
"""
num_classse = 2
model = MyModel(num_classse)


model.compile(
    optimizer="adam",
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.build((None, 256, 256, 3))  
model.summary()

checkpoint_path = "modelCheckpoint.h5"


"""
learning_rate_reduction = ReduceLROnPlateau(
    monitor = 'val_accuracy',
    patience=2,
    factor=0.5,
    min_lr = 0.0005,
    verbose = 1
)

early_stoping = EarlyStopping(
    monitor='val_loss',
    patience= 5, 
    restore_best_weights=True,
    verbose=0
)
loaded_model = tf.keras.models.load_model('full_model_subclass2')
his = loaded_model.fit(
    training_data, 
    validation_data=validation_data, 
    epochs=70,
    callbacks=[early_stoping, learning_rate_reduction],
)

loaded_model.save('full_model_subclass3', save_format='tf')

with open('training_history3.pkl', 'wb') as file:
    pickle.dump(his, file)
