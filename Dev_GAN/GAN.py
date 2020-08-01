# Numpy and matplotlib
import numpy as np
import matplotlib.pyplot as plt

# Dataset
import keras.datasets as ds

# Unconditional GAN
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, BatchNormalization

# Conditional GAN
from keras.models import Model
from keras.layers import Input, Embedding, Concatenate

# Visualization tools
from IPython.display import clear_output, Image



# Load selected dataset
def load_dataset(dataset):

	if dataset == 'MNIST':
		(trainX, trainy), (_, _) = ds.mnist.load_data()

	if dataset == 'Fashion-MNIST':
		(trainX, trainy), (_, _) = ds.fanshion_mnist.load_data()

	# Exapnd to 3D adding one channel
	X = np.expand_dims(trainX, axis=-1)
		
	# Convert from int to float and rescale from [0, 255] to [-1, 1]
	X = X.astype('float32')
	X = (X - (255 / 2)) / (255 / 2)
	return [X, trainy]


class DCGAN:

	# Initialization function
	def __init__(self, dataset):

		[trainX, trainy] = load_dataset(dataset)
		self.img_size = self.trainX[0].shape[0]


		#self.discriminator_path = discriminator_path
		#self.generator_path = generator_path
		#self.output_directory = output_directory


	def show_dataset(self):
		print('Train: ', self.trainX.shape)


	# Build Discriminator
	def build_discriminator(self):

		input_shape = (img_size, img_size, 1)

		# Initialize the NN
		model = Sequential()

		# First convolutional layer
		model.add(Conv2D(64, 3, strides=2, padding='same', input_shape=input_shape))
		model.add(BatchNormalization(momentum=0.9))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.4))

		# Second convolutional layer
		model.add(Conv2D(128, 3, strides=2, padding='same'))
		model.add(BatchNormalization(momentum=0.9))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dropout(0.4))

		# Flattening and output layer
		model.add(Flatten())
		model.add(Dense(1, activation='sigmoid'))

		opt = Adam(lr=0.0002, beta_1=0.5)
		model.compile(loss='binary_crossentropy', optimization=opt, metrics=['accuracy'])
		
		return model


	# Build Generator with default latent_space=100
	def build_generator(self, latent_dim=100):

		# Initialize the NN
		model = Sequential()

		# Fully connected layer
		model.add(Dense(7 * 7 * 256, input_dim=latent_dim))
		model.add(BatchNormalization(momentum=0.9))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Reshape(7, 7, 256))
		model.add(Dropout(0.4))

		# First upsampling layer 14x14
		model.add(Upsampling2D())
		model.add(Conv2D(128, 3, padding='same'))
		model.add(BatchNormalization(momentum=0.9))
		model.add(LeakyReLU(alpha=0.2))

		# Second upsampling layer 28x28
		model.add(Upsampling2D())
		model.add(Conv2D(128, 3, padding='same'))
		model.add(BatchNormalization(momentum=0.9))
		model.add(LeakyReLU(alpha=0.2))

		# Output layer
		model.add(Conv2D(1, 7, activation='tanh', padding='same'))

		return model


	# Build the GAN framework
	def build_gan(self):

		opt = Adam(lr=0.0002, beta_1=0.5)

		# Build Discriminator and Generator
		self.discriminator = self.build_discriminator()
		self.generator = self.build_generator()
		self.generator.compile(loss='binary_crossentropy', optimization=opt)

		# Freeze discriminator weights during generator training
		self.discriminator.trainable = False

		# Connect generator and discriminator
		self.gan = Sequential()
		self.gan.add(self.generator)
		self.gan.add(self.discriminator)
		self.gan.compile(loss='binary_crossentropy', optimization=opt)


	'''
	# Train the GAN
	def train(self, epochs=20, batch_size=128):

		self.build_gan()

		batch_per_epoch = int(self.trainX.shape[0] / batch_size)
		half_batch = int(batch_per_epoch / 2)


		# Loop over epochs
		for epoch in range(epochs):

			for mbatch in range(batch_per_epoch):

				# Random select half_batch real samples


				# Random generate half_batch fake samples


				# Stacks all samples together and train the discriminator


				# Prepare a batch_size random noise for the generator + labels

				# Train the generator
	'''


