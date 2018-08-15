import csv
import cv2
import os
import numpy as np
import sklearn

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers import Cropping2D
from keras.utils.visualize_util import plot

import matplotlib.pyplot as plt

def plot_error(history):
	print('Inside plot')
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model mean squared error loss')
	plt.ylabel('mean squared error loss')
	plt.xlabel('epoch')
	plt.legend(['training set', 'validation set'], loc='upper right')
	plt.show()

def generator(samples, batch_size=128):
	num_samples = len(samples)
	while 1:
		sklearn.utils.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			angles = []
			index = 0
			for batch_sample in batch_samples:
				index = index+1
				name = 'data/IMG/' + batch_sample[0].split('/')[-1]
				image = cv2.imread(name)
				image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
				angle = float(batch_sample[1])
				"""if (index % 8 == 0):
					flipped_image = cv2.flip(image, 1)
					flipped_angle = angle*-1.0
					images.append(flipped_image)
					angles.append(flipped_angle)
				else:"""
				images.append(image)
				angles.append(angle)
								
		X_train = np.array(images)
		y_train = np.array(angles)
		yield sklearn.utils.shuffle(X_train, y_train)

def get_model():
	print('Inside model()')
	model = Sequential()
	model.add(Lambda(lambda x: x /255.0 - 0.5, input_shape=(160, 320, 3)))
	model.add(Cropping2D(cropping=((55,25),(0,0))))
	model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
	model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
	model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
	model.add(Convolution2D(64,3,3,activation="relu"))
	model.add(Convolution2D(64,3,3,activation="relu"))
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))
	model.compile(loss='mse',optimizer='adam')
	return model

def train():
	samples = []
	with open('data/driving_log.csv') as csvfile:

		reader = csv.reader(csvfile)
		csvfile.readline()
		correction = [0.0, 0.25, -0.25]
		for line in reader:
			for i in range(3):
				angle = float(line[3]) + correction[i]
				new_sample = (line[i],angle)
				samples.append(new_sample)

	train_samples, validation_samples = train_test_split(samples, test_size=0.2)

	train_generator = generator(train_samples, batch_size=128)
	validation_generator = generator(validation_samples, batch_size=128)

	model = get_model()

	plot(model, to_file='model.png', show_shapes=True, show_layer_names=False)
	history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=2, verbose=1)
	model.save('model.h5')

	plot_error(history_object)

if __name__ == '__main__':
	print('Trainning started')
	train()
	print('Trainning complete!')