import os
import numpy as np
from PIL import Image


training_data = []


for filename in os.listdir('Agnes_Lawrence_Pelton_pictures/'):
    path = os.path.join('Agnes_Lawrence_Pelton_pictures/', filename)
    image = Image.open(path).resize((128, 128), Image.ANTIALIAS)

    training_data.append(np.asarray(image))

training_data = np.reshape(training_data, (-1, 128, 128, 3))

training_data = training_data / 127.5 - 1

np.save('Agnes_Lawrence_Pelton_pictures.npy', training_data)