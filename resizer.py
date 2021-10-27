import os
import numpy as np
from PIL import Image


training_data = []


for filename in os.listdir('henri_matisse_abstract_expressionism/'):
    path = os.path.join('henri_matisse_abstract_expressionism/', filename)
    image = Image.open(path).resize((128, 128), Image.ANTIALIAS)

    training_data.append(np.asarray(image))

training_data = np.reshape(training_data, (-1, 128, 128, 3))

training_data = training_data / 127.5 - 1

np.save('henri_matisse_abstract_expressionism.npy', training_data)