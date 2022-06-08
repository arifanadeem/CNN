from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import tensorflow as tf


datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

img = tf.keras.preprocessing.image.load_img('C:\\Users\\arifa\\Downloads\\MACHINE LEARNING\\MACHINE LEARNING\\deep_learning\\CNN\\bird.jpg')  # this is a PIL image

x = tf.keras.preprocessing.image.img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='C:\\Users\\arifa\\Downloads\\MACHINE LEARNING\\MACHINE LEARNING\\deep_learning\\preview', save_prefix='cat', save_format='jpeg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely