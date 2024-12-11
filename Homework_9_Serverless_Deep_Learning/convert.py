import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('model_2024_hairstyle.keras')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
 
with open('model_2024_hairstyle.tflite', 'wb') as f_out:
    f_out.write(tflite_model)