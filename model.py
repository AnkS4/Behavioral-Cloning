import csv
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Conv2D, MaxPooling2D, Cropping2D
from keras import backend as K

lines = []
with open('./data/driving_log.csv') as f:
	reader = csv.reader(f)
	for line in reader:
		lines.append(line)

images = []
measurements = []
correction = 0.2
for line in lines:
	for i in range(3):
		source_path = line[i]
		filename = source_path.split('/')[-1]
		local_path = './data/IMG/' + filename
		image = cv2.imread(local_path)
		#image = cv2.resize(image, (32, 32))
		#image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		images.append(image)
	measurement = float(line[3])
	measurements.append(measurement)
	measurements.append(measurement+correction)
	measurements.append(measurement-correction)

print(len(images))

#conv = np.array([[0.2989],[0.5870],[0.1140]])
def grayconv(img):
	'''
	img2 = np.zeros((img.shape[0],img.shape[1],1))
	img2[:,:,:,] = np.dot(img[...,:3],conv)
	return img2
	'''
	return (0.21 * img[:,:,:1]) + (0.72 * img[:,:,1:2]) + (0.07 * img[:,:,-1:])


augmented_images, augmented_measurements= [], []
for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_images.append(cv2.flip(image, 1))
	augmented_measurements.append(measurement)
	augmented_measurements.append(-1.0 * measurement)


X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

'''
X_train = np.array(images)
y_train = np.array(measurements)
'''
print(X_train.shape)

'''
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((50, 20), (0, 0))))
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(64, (3, 3), strides=(2, 2), activation='relu'))
model.add(Conv2D(64, (3, 3), strides=(2, 2), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
'''

model = Sequential()
model.add(Lambda(lambda x: grayconv(x), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x/255.0 - 0.5))
model.add(Cropping2D(cropping=((50, 20), (0, 0))))

model.add(Lambda(lambda image:K.tf.image.resize_images(image, size=(32, 32))))
model.add(Conv2D(32, (3, 3), strides=(2, 2), activation='elu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(12, (3, 3), strides=(2, 2), activation='elu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(30))
model.add(Dense(5))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3)
model.save('model.h5')
