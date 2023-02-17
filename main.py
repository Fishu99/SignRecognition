import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import pandas as pd

# Initialization parameters
storedTrainPath = "TrainData"   # Catalog with data for training
storedTestPath = "TestData"     # Catalog with data for testing
storedValidationPath = "ValidationData"     # Catalog with data for validating
labelsCsv = "labels.csv"  # File with sign labels
modelOutput = "Model_03"
imgDims = (32, 32, 3)   # Training Image dimensions
epochsCount = 20 # Amount of epochs
stepsPerEpoch = 250     # Number of steps in each epoch of training - should be trainX/batchSize
batchSize = 50  # Number of elements to process together

numberOfClasses = len(os.listdir(storedTrainPath))
# Arrays for sets data
trainX = []
trainY = []
testX = []
testY = []
validationX = []
validationY = []
# Traffic sign class counter
classCounter = 0
# Download images for every traffic sign class
for i in range(0, numberOfClasses):
    # Get traffic sign class directory for every set
    imgTrainList = os.listdir(storedTrainPath + "/" + str(classCounter))
    imgTestList = os.listdir(storedTestPath + "/" + str(classCounter))
    imgValidationList = os.listdir(storedValidationPath + "/" + str(classCounter))
    for j in imgTrainList:
        # Get image from train set directory
        currentImage = cv2.imread(storedTrainPath + "/" + str(classCounter) + "/" + j)
        trainX.append(currentImage)
        trainY.append(classCounter)
    for j in imgTestList:
        # Get image from test set directory
        currentImage = cv2.imread(storedTestPath + "/" + str(classCounter) + "/" + j)
        testX.append(currentImage)
        testY.append(classCounter)
    for j in imgValidationList:
        # Get image from validation set directory
        currentImage = cv2.imread(storedValidationPath + "/" + str(classCounter) + "/" + j)
        validationX.append(currentImage)
        validationY.append(classCounter)
    print(classCounter, end=" ")
    classCounter += 1
print(" ")


trainX = np.array(trainX)
trainY = np.array(trainY)
testX = np.array(testX)
testY = np.array(testY)
validationX = np.array(validationX)
validationY = np.array(validationY)

# Print sets shapes
print("Data Shapes")
print("Train", end="")
print(trainX.shape, trainY.shape)
print("Validation", end="")
print(validationX.shape, validationY.shape)
print("Test", end="")
print(testX.shape, testY.shape)

# Validate sets data
assert(trainX.shape[0] == trainY.shape[0]), "The number of images in not equal to the number of labels in training set"
assert(validationX.shape[0] == validationY.shape[0]), "The number of images in not equal to the number of labels"
assert(testX.shape[0] == testY.shape[0]), "The number of images in not equal to the number of labels in test set"
assert(trainX.shape[1:] == imgDims), " The dimensions of the Training images are wrong "
assert(validationX.shape[1:] == imgDims), " The dimensions of the Validation images are wrong "
assert(testX.shape[1:] == imgDims), " The dimensions of the Test images are wrong"

# Reading traffic sign classes names form lables.csv file
dataFromCSV = pd.read_csv(labelsCsv)  # Read data from labels csv file
print("Data shape ", dataFromCSV.shape, type(dataFromCSV))  # Save Name column values to array

# Show number of samples for each class
numberOfSamples = []

for curClassIndex, row in dataFromCSV.iterrows():
    x_selected = trainX[trainY == curClassIndex]
    numberOfSamples.append(len(x_selected))

# Printing plot with number of samples for each class
print(numberOfSamples)
plt.figure(num=modelOutput + " - Distribution", figsize=(12, 4))
plt.bar(range(0, numberOfClasses), numberOfSamples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()


# Function that applies grayscale, standardize the lighting and normalize values
def prepareImage(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)     # Convert  image to grayscale
    img = cv2.equalizeHist(img)      # Standardize the lighting in image
    img = img/255            # Normalize values to 0-1
    return img


# Prepare images in every set array
print("Preparing images in arrays")
trainX = np.array(list(map(prepareImage, trainX)))
validationX = np.array(list(map(prepareImage, validationX)))
testX = np.array(list(map(prepareImage, testX)))

# Adding depth of 1 to data
trainX = trainX.reshape(trainX.shape[0], trainX.shape[1], trainX.shape[2], 1)
validationX = validationX.reshape(validationX.shape[0], validationX.shape[1], validationX.shape[2], 1)
testX = testX.reshape(testX.shape[0], testX.shape[1], testX.shape[2], 1)

# Augmentation for training data
dataGen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    rotation_range=10
)
dataGen.fit(trainX)
batches = dataGen.flow(trainX, trainY, batch_size=20)
batchX, batchY = next(batches)

# Making Y arrays categorical
trainY = to_categorical(trainY, numberOfClasses)
validationY = to_categorical(validationY, numberOfClasses)
testY = to_categorical(testY, numberOfClasses)


# Convolution neural network model
def myModel():
    numberOfFilters = 60  # Number of filters for convolution
    sizeOfFilter = (5, 5)  # Size of kernel for beginning convolution layer instances
    sizeOfFilter2 = (3, 3)  # Size of kernel for next convolution layers
    sizeOfPool = (2, 2)  # Reduce overfitting
    numberOfNodes = 500  # Number of nodes in hidden layer
    model = Sequential()  # Set model as sequential
    # Add convolution layers
    model.add((Conv2D(numberOfFilters, sizeOfFilter, input_shape=(imgDims[0], imgDims[1], 1), activation='relu')))
    model.add((Conv2D(numberOfFilters, sizeOfFilter, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(numberOfFilters // 2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(numberOfFilters // 2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))  # Inputs nodes to drop with each update -> 1/all 0/none
    model.add(Flatten())  # Add Flatten layer to set output data format
    # Add dense layers for final classification
    model.add(Dense(numberOfNodes, activation='relu'))
    model.add(Dropout(0.5))  # Inputs nodes to drop with each update -> 1/all 0/none
    model.add(Dense(numberOfClasses, activation='softmax'))  # Output layer
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])  # Compile model
    return model


# Creating model
model = myModel()
print(model.summary())
# Create model history for plot that will be created
history = model.fit_generator(
    dataGen.flow(trainX, trainY, batch_size=batchSize),
    steps_per_epoch=stepsPerEpoch,
    epochs=epochsCount,
    validation_data=(validationX,validationY),
    shuffle=1
)

# Model history plot
plt.figure(num=modelOutput + " - Loss")
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epoch')

plt.figure(num=modelOutput + " - Accuracy")
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')

plt.show()
score = model.evaluate(testX, testY, verbose=0)
print('Test Score:', score[0])
print('Test Accuracy:', score[1])

# Save model
model.save(modelOutput)
