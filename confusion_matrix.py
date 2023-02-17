import itertools
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.metrics import confusion_matrix

# Import model
print("Loading model")
modelName = 'Model_03'
model = tf.keras.models.load_model(modelName)
print("Model loaded")
# Import data from csv
labelsCsv = "labels.csv"  # File with sign labels
dataFromCsv = pd.read_csv(labelsCsv)  # Read data from labels csv file
labels = dataFromCsv['Name'].to_numpy()  # Save Name column values to array

print("Downloading images")
testX = []
testY = []
storedTestPath = "TestData"     # Catalog with data for testing
numberOfClasses = len(os.listdir(storedTestPath))
# Traffic sign class counter
classCounter = 0
# Download images for every traffic sign class
for i in range(0, numberOfClasses):
    # Get traffic sign class directory
    imgTestList = os.listdir(storedTestPath + "/" + str(classCounter))
    for j in imgTestList:
        # Get image from directory
        currentImage = cv2.imread(storedTestPath + "/" + str(classCounter) + "/" + j)
        testX.append(currentImage)
        testY.append(classCounter)
    print(classCounter, end=" ")
    classCounter += 1
print(" ")

testX = np.array(testX)
testY = np.array(testY)

# Method creates confusion matrix plot from given data
def plotConfusionMatrix(cm, classes, cmap=plt.cm.Blues):
    # Plot confusion matrix
    pltName = modelName + '_CMplot'
    px = 1 / plt.rcParams['figure.dpi']
    plt.figure(num=pltName, figsize=(1680 * px, 1050 * px))

    # Display results as percentage values
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = cm * 100
    # Setting plot attributes
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))

    shortenClasses = classes
    for i in range(0, len(classes)):
        shortenClasses[i] = classes[i][0:26]

    plt.xticks(tick_marks, shortenClasses, rotation='vertical')
    plt.yticks(tick_marks, shortenClasses)

    fmt = '.0f'  # Set format
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j] > 0:
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     verticalalignment = "center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()  # Adjusting the padding in the plot figures
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.grid(color='cornflowerblue', linestyle=':', linewidth=0.5)  # Setting plot style
    plt.grid(True)  # Show grid
    plt.subplots_adjust(bottom=0.25, left=0.2, top=0.95, right=0.8)  # Adjusting the spacing for subplots
    plt.savefig(pltName + '.png')  # Saving the current figure.
    # Maximize plot window
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    # Show plot on the screen
    plt.show()


# Method to process image for predictions
def prepareImage(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)     # Convert  image to grayscale
    img = cv2.equalizeHist(img)  # Standardize the lighting in image
    img = img/255  # Normalize values to 0-1
    return img


print("Preparing images in arrays")
testX = np.array(list(map(prepareImage, testX)))
testX = testX.reshape(testX.shape[0], testX.shape[1], testX.shape[2], 1)

# Compute confusion matrix
predictions = model.predict(testX)  # Make predictions
predY = np.argmax(predictions, axis=1)
confusionMatrix = confusion_matrix(testY, predY)
np.set_printoptions(precision=2)  # Set NumPy to 2 decimal places
plotConfusionMatrix(confusionMatrix, classes=labels)