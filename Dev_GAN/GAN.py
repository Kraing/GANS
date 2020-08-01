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


class GAN:

	# Initialization function
	def __init__(self, discriminator_path, generator_path, output_directory, img_size):
    	self.img_size = img_size
    	self.discriminator_path = discriminator_path
    	self.generator_path = generator_path
    	self.output_directory = output_directory


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

