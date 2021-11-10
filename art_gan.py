# https://towardsdatascience.com/generating-modern-arts-using-generative-adversarial-network-gan-on-spell-39f67f83c7b4
# https://www.wga.hu/index1.html

from tensorflow import keras

from tensorflow.keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation, ZeroPadding2D, Conv2D, UpSampling2D, LeakyReLU
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
from PIL import Image
import os
import statistics

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.333
session = InteractiveSession(config=config)

# Preview image Frame (output images eachs number of iterations)
PREVIEW_ROWS = 4
PREVIEW_COLS = 7
PREVIEW_MARGIN = 4
SAVE_FREQ = 100

# Size vector to generate images from (generator input)
NOISE_SIZE = 50

# Configuration
EPOCHS = 10000 # number of iterations
BATCH_SIZE = 1 # -> default 32
GENERATE_RES = 3 # add layers to the generator (generate resolution) -> default 3
IMAGE_SIZE = 128 # rows/cols
IMAGE_CHANNELS = 3

training_data = np.load('wga.npy')

def build_discriminator(image_shape):
# The discriminator is a binary classifier. It needs to classify either the data is real or fake
# Input = image (true & false)
# Output = probability to be true or false (sigmoid)
	model = Sequential()

	model.add(Conv2D(32, kernel_size=3, strides=2,
		input_shape=image_shape, padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.25))
	model.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))
	model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
	model.add(BatchNormalization(momentum=0.8))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.25))
	model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
	model.add(BatchNormalization(momentum=0.8))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.25))
	model.add(Conv2D(256, kernel_size=3, strides=1, padding='same'))
	model.add(BatchNormalization(momentum=0.8))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.25))
	model.add(Conv2D(512, kernel_size=3, strides=1, padding='same'))
	model.add(BatchNormalization(momentum=0.8))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))

	model.summary()

	input_image = Input(shape=image_shape)
	validity = model(input_image)

	return Model(input_image, validity)




def build_generator(noise_size, channels):
# The generator is responsible for generating different kinds of noise data
# Input = noise (noise_shape vector with random values)
# Output = generated image 
	model = Sequential()

	model.add(Dense(4 * 4 * 256, activation='relu',input_dim=noise_size))
	model.add(Reshape((4, 4, 256)))
	model.add(UpSampling2D())
	model.add(Conv2D(256, kernel_size=3, padding='same'))
	model.add(BatchNormalization(momentum=0.8))
	model.add(Activation('relu'))
	model.add(UpSampling2D())
	model.add(Conv2D(256, kernel_size=3, padding='same'))
	model.add(BatchNormalization(momentum=0.8))
	model.add(Activation('relu'))

	for i in range(GENERATE_RES):
		model.add(UpSampling2D())
		model.add(Conv2D(256, kernel_size=3, padding='same'))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Activation('relu'))

	model.add(Conv2D(channels, kernel_size=3, padding='same'))
	model.add(Activation('tanh'))

	model.summary()

	input = Input(shape=(noise_size,))
	generated_image = model(input)

	return Model(input, generated_image)




def save_images(cnt, noise):
	# create a shape to print the output
	image_array = np.full((
		PREVIEW_MARGIN + (PREVIEW_ROWS * (IMAGE_SIZE + PREVIEW_MARGIN)),
		PREVIEW_MARGIN + (PREVIEW_COLS * (IMAGE_SIZE + PREVIEW_MARGIN)), 3),
	255, dtype=np.uint8)
	# generate image from a random noise to see the network progress
	generated_images = generator.predict(noise) # Generate 28 images, why 28 ? shape=(28,128,128,3)

	# normalized data
	generated_images = 0.5 * generated_images + 0.5

	# for the number of image to output print the generated images
	image_count = 0
	for row in range(PREVIEW_ROWS):
		for col in range(PREVIEW_COLS):
			r = row * (IMAGE_SIZE + PREVIEW_MARGIN) + PREVIEW_MARGIN
			c = col * (IMAGE_SIZE + PREVIEW_MARGIN) + PREVIEW_MARGIN
			image_array[r:r + IMAGE_SIZE, c:c + IMAGE_SIZE] = generated_images[image_count] * 255
			image_count += 1
	
	output_path = 'output'
	if not os.path.exists(output_path):
		os.makedirs(output_path)

	filename = os.path.join(output_path, f"trained-{cnt}.png")
	im = Image.fromarray(image_array)
	im.save(filename)



#input shape of images
image_shape = (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)

optimizer = Adam(1.5e-4, 0.5)

#build discriminator with input shape
discriminator = build_discriminator(image_shape)
discriminator.compile(loss='binary_crossentropy',optimizer=optimizer, metrics=['accuracy'])

# build generator with input shape and link it random noise output
generator = build_generator(NOISE_SIZE, IMAGE_CHANNELS)
random_input = Input(shape=(NOISE_SIZE,))
generated_image = generator(random_input)

# since we are only training generators here, we do not want to adjust the weights of the discriminator
discriminator.trainable = False

# link the discriminator to generator output
validity = discriminator(generated_image)

# Combined model
# Input = random noise 
# Output = validity, is it a real picture ?
combined = Model(random_input, validity)
combined.compile(loss='binary_crossentropy',optimizer=optimizer, metrics=['accuracy'])

# create batch_size n list of 1's and 0's
y_real = np.ones((BATCH_SIZE, 1))
y_fake = np.zeros((BATCH_SIZE, 1))

# create a constant noise to inject in the generator to get an output for the saving image function
fixed_noise = np.random.normal(0, 1, (PREVIEW_ROWS * PREVIEW_COLS, NOISE_SIZE))

discriminator_accuracy_list = []
generator_accuracy_list = []

cnt = 1
for epoch in range(EPOCHS):
	# get some random real data
	idx = np.random.randint(0, training_data.shape[0], BATCH_SIZE)
	x_real = training_data[idx]

	# get some random false data
	noise= np.random.normal(0, 1, (BATCH_SIZE, NOISE_SIZE))
	x_fake = generator.predict(noise)

	# train on batch the discriminator
	# some research has shown that training them separately (real and fake data) can get us some better results
	discriminator_metric_real = discriminator.train_on_batch(x_real, y_real)
	discriminator_metric_generated = discriminator.train_on_batch(x_fake, y_fake)

	# compute the means of the metrics
	discriminator_metric = 0.5 * np.add(discriminator_metric_real, discriminator_metric_generated)

	# train on batch the generator
	generator_metric = combined.train_on_batch(noise, y_real)

	#means of all the epochs
	discriminator_accuracy_list.append(100*discriminator_metric[1])
	generator_accuracy_list.append(100*generator_metric[1])

	mean_discriminator_accuracy = statistics.mean(discriminator_accuracy_list)
	mean_generator_accuracy = statistics.mean(generator_accuracy_list)


	# save output of the generator with random noise every SAVE_FREQ n epochs and display accuracy
	if epoch % SAVE_FREQ == 0:
		save_images(cnt, fixed_noise)
		cnt += 1
		print(f'Mean discriminator accuracy: {mean_discriminator_accuracy}, Mean generator accuracy: {mean_generator_accuracy}')
		print(f'{epoch} epoch, Discriminator accuracy: {100*  discriminator_metric[1]}, Generator accuracy: {100 * generator_metric[1]}')