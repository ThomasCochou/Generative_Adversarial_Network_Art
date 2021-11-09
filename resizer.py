import os
import numpy as np
from PIL import Image


training_data = []

input_path = "wga"

i = 0
for filename in os.listdir(input_path+"/"):
    path = os.path.join(input_path+"/", filename)
    image = Image.open(path).resize((128, 128), Image.ANTIALIAS)

    training_data.append(np.asarray(image))
    print(str(i), end="\r")
    i = i + 1

training_data = np.reshape(training_data, (-1, 128, 128, 3))

training_data = training_data / 127.5 - 1

np.save(input_path+".npy", training_data)