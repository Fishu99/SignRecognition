import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split

# COLLECTING DATA, SPLITTING IT (TRAIN, TEST, VALIDATION) AND SAVE IT TO APPROPRIATE NEWLY CREATED FOLDERS

#### Initialization parameters ####
imagesPath = "OriginalData"  # Catalog with data
storedTrainPath = "TrainData"   # Catalog with data for training
storedTestPath = "TestData"     # Catalog with data for testing
storedValidationPath = "ValidationData"     # Catalog with data for validating
labelsCsv = "labels.csv"  # File with sign labels
imgDims = (32, 32, 3)   # Training Image dimensions
testRatio = 0.2     # It represents the proportion of the dataset to include in the test split
validationRatio = 0.2   # It represents the proportion of the dataset to include in the validation split

#### Loading images ####
classCounter = 0    # Represents class of the sign
images = []     # Stores all the images
imagesLabels = []   # Stores all IDs of upper images
dirList = os.listdir(imagesPath)
signsClassesCount = len(dirList)
print("Detected : ", signsClassesCount, " catalogs with signs.")
print("Downloading signs data")
for i in range(0, len(dirList)):
    imgDirsList = os.listdir(imagesPath + "/" + str(classCounter))
    for j in imgDirsList:
        currentImage = cv2.imread(imagesPath + "/" + str(classCounter) + "/" + j)
        images.append(currentImage)
        imagesLabels.append(classCounter)
    print(classCounter, end=" ")
    classCounter += 1
print(" ")
images = np.array(images)
imagesLabels = np.array(imagesLabels)

#### Split data to train, test and validation arrays ####
# trainX = Array of images for model training
# trainY = Sign class ids
print("Creating arrays")
trainX, testX, trainY, testY = train_test_split(images, imagesLabels, test_size=testRatio)
trainX, validationX, trainY, validationY = train_test_split(trainX, trainY, test_size=validationRatio)

#### Checking if number of images is equal to number of image labels ####
print("Data Shapes")
print("Train", end="")
print(trainX.shape, trainY.shape)
print("Validation", end="")
print(validationX.shape, validationY.shape)
print("Test", end="")
print(testX.shape, testY.shape)
assert(trainX.shape[0] == trainY.shape[0]), "The number of images in not equal to the number of labels in training set"
assert(validationX.shape[0] == validationY.shape[0]), "The number of images in not equal to the number of labels"
assert(testX.shape[0] == testY.shape[0]), "The number of images in not equal to the number of labels in test set"
assert(trainX.shape[1:] == imgDims), " The dimensions of the Training images are wrong "
assert(validationX.shape[1:] == imgDims), " The dimensions of the Validation images are wrong "
assert(testX.shape[1:] == imgDims), " The dimensions of the Test images are wrong"

#### Writing images ####

#Things that must be saved:
    #trainX, trainY
    #testX, testY
    #validationX, validationY

classAmount = classCounter
classCounter = 0
try:
    for i in range(0, classAmount):
        #Creating directiories
        dirTrainName = storedTrainPath + "/" + str(classCounter)
        dirTestName = storedTestPath + "/" + str(classCounter)
        dirValidationName = storedValidationPath + "/" + str(classCounter)
        os.makedirs(dirTrainName)
        os.makedirs(dirTestName)
        os.makedirs(dirValidationName)
        classCounter += 1


    #Saving splitted data into them
    classOccurTrainCounter = [0] * classAmount
    classOccurTestCounter = [0] * classAmount
    classOccurValidationCounter = [0] * classAmount
    savedFilesCounter = 0
    #TrainX, TrainY
    for j in range(0, trainX.shape[0]):
        savingCurrFilePath = storedTrainPath + "/" + str(trainY[j]) + "/" + str(classOccurTrainCounter[trainY[j]]) + ".jpg"
        cv2.imwrite(savingCurrFilePath, trainX[j])
        savedFilesCounter += 1
        classOccurTrainCounter[trainY[j]] += 1

    print("- TRAIN data: saved files + " + str(savedFilesCounter))
    savedFilesCounter = 0

    #TestX, TestY
    for j in range(0, testX.shape[0]):
        savingCurrFilePath = storedTestPath + "/" + str(testY[j]) + "/" + str(classOccurTestCounter[testY[j]]) + ".jpg"
        cv2.imwrite(savingCurrFilePath, testX[j])
        savedFilesCounter += 1
        classOccurTestCounter[testY[j]] += 1

    print("- TEST data: saved files + " + str(savedFilesCounter))
    savedFilesCounter = 0

    #ValidationX, ValidationY
    for j in range(0, validationX.shape[0]):
        savingCurrFilePath = storedValidationPath + "/" + str(validationY[j]) + "/" + str(classOccurValidationCounter[validationY[j]]) + ".jpg"
        cv2.imwrite(savingCurrFilePath, validationX[j])
        savedFilesCounter += 1
        classOccurValidationCounter[validationY[j]] += 1

    print("- VALIDATION data: saved files + " + str(savedFilesCounter))
    savedFilesCounter = 0

except FileExistsError:
    print("Directory ", storedTestPath,  " already exists")