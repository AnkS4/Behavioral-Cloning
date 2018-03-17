import csv
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Conv2D, MaxPooling2D, Cropping2D, Dropout, LeakyReLU
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras import optimizers

from keras.models import load_model
model = load_model('model.h5')

lines = []
with open('./Car_Train7/driving_log2.csv') as f:
	reader = csv.reader(f)
	for line in reader:
		lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

correction = 0.2
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
            	path = './Car_Train7/IMG/' + batch_sample[0].split('/')[-1]
            	image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2HSV)
            	tmp = image[:,:,1].reshape(160, 320, 1)
            	images.append(tmp)
            	#tmp2 = cv2.flip(tmp,1).reshape(160, 320, 1)
            	#images.append(tmp2)

            	angle = float(batch_sample[3])
            	#temp = [angle, -angle]
            	angles.append(angle)
            	#angles.extend(temp)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

model.compile(optimizer='Adam', loss='mse')
model.fit_generator(train_generator, steps_per_epoch= len(train_samples), validation_data=validation_generator, validation_steps=len(validation_samples), epochs=5, verbose=1)
model.save('model2.h5')