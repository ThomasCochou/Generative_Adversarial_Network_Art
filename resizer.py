import os
import numpy as np
from PIL import Image


training_data = []

input_path = "wga"

for filename in os.listdir(input_path+"/"):
    path = os.path.join(input_path+"/", filename)
    image = Image.open(path).resize((128, 128), Image.ANTIALIAS)

    training_data.append(np.asarray(image))

training_data = np.reshape(training_data, (-1, 128, 128, 3))

training_data = training_data / 127.5 - 1

np.save(input_path+".npy", training_data)