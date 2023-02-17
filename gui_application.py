import cv2
import numpy
import tensorflow as tf
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

font = cv2.FONT_HERSHEY_SIMPLEX
# Import model
print("Loading model")
model = tf.keras.models.load_model('Model_02')
print("Model loaded")
# Import data from csv
labelsCsv = "labels.csv"  # File with sign labels
dataFromCsv = pd.read_csv(labelsCsv)  # Read data from labels csv file
labels = dataFromCsv['Name'].to_numpy()  # Save Name column values to array


# Method to process image for predictions
def prepareImage(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert  image to grayscale
    img = cv2.equalizeHist(img)  # standardize the lighting in image
    img = img / 255  # normalize values to 0-1
    return img


# Method to get traffic sign class name that was read from .csv file
def getClassName(classNo):
    return labels[classNo]


# Prints probability stats - version with no sorted parameter
def printProbabilityInfoNoSort(predictions):
    print("\n_____________\n")
    for i in range(0, len(predictions[0])):
        outputPercentage = "{:.4f}". format(predictions[0][i] * 100)
        outputClassName = getClassName(i)[0:26]
        if outputPercentage != "0.0000":
            print (str(i) + '.\t' + outputPercentage + '% - ' + outputClassName)
    print("\n_____________\n")


# Prints probability stats - version with sorted parameters
def printProbabilityInfoSort(predictions0, classes):
    assert (len(predictions0) == len(classes)), "The number of images in not equal to the number of lables in training set"
    length = len(predictions0)
    print("\n_____________\n")
    for i in range(0, length):
        outputPercentage = "{:.4f}". format(predictions0[i] * 100)
        outputClassName =  getClassName(classes[i])[0:26] + " -> [id_"+ str(classes[i]) + "] "
        if outputPercentage != "0.0000":
            print (str(i) + '.\t' + outputPercentage + '% - ' + outputClassName)
    print("\n_____________\n")


# Method that sorts prediction list
def getSortedPredictionList(predictions):
    classesAmount = len(predictions[0])
    classes = np.arange(start=0, stop=classesAmount, step=1)
    permutation = (np.argsort(predictions[0]))[::-1]

    predictionsSorted = predictions[0][permutation]
    classes = classes[permutation]
    return predictionsSorted, classes


# Method that returns shorted (less nb. of signs in the string) names of traffic sign classes
def shortenClassLabels(classes, nbSigns):
    shorted = []
    for i in classes:
        shorted.append(i[0:nbSigns])
    classes = np.array(shorted)
    return classes


# Method for printing a selection probability histogram on the screen
def makeAProbabilityPlot(predictions0, classes):
    # Set plot attributes for probability histogram
    px = 1 / plt.rcParams['figure.dpi']
    plt.figure(num="Recognition - probability plot", figsize=(480*px, 640*px))
    plt.clf()   # Removes previous data from plot without closing it
    plt.bar(range(0, len(classes)), predictions0)
    plt.title("Probability of choosing a class")
    plt.xlabel("Class name")
    plt.ylabel("Percentage")
    nbOfSigns = 21
    classes = shortenClassLabels(classes, nbOfSigns)
    plt.xticks(range(0, len(classes)), classes, rotation='vertical')
    plt.xlim([0, 10])
    plt.subplots_adjust(bottom=0.3)
    plt.show()


# Get labels for traffic sign classes from .csv file values
def getSortedLabelsForPlot(classes):
    sortedLabels = []
    for i in range(0, len(classes)):
        sortedLabels.insert(i, labels[classes[i]])
    return numpy.array(sortedLabels)


def openImage():
    root.withdraw()
    # Get filename form file dialog
    root.filename = filedialog.askopenfilename(
        title = "Select File",
        filetypes = [
            ('image files', '.jpg')
        ])

    # Check if a file was selected
    if not root.filename:
        print("No file selected.")
        root.deiconify()
        return

    root.deiconify()

    imageLabel.config(text=root.filename)
    # Read and process selected file
    imgOriginal = cv2.imread(root.filename)
    b, g, r = cv2.split(imgOriginal)
    imgOriginal = cv2.merge((r, g, b))
    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (32, 32))
    img = prepareImage(img)
    img = img.reshape(1, 32, 32, 1)

    # Make predictions for selected file
    predictions = model.predict(img)
    classIndex = np.argmax(predictions) #model.predict_classes(img)
    probabilityValue = np.amax(predictions)

    # Print prediction statistics
    predictionsSorted, classes = getSortedPredictionList(predictions)
    sortedLabels = getSortedLabelsForPlot(classes)
    printProbabilityInfoSort(predictionsSorted, classes)

    classLabel.config(text=str(classIndex) + ' ' + str(getClassName(classIndex)))
    probabilityLabel.config(text=str(round(probabilityValue*100, 2)) + "%")

    # Show image to user
    aspectRatioY = imgOriginal.shape[1] / imgOriginal.shape[0]
    widthX = 256
    widthY = int(widthX / aspectRatioY)
    imgToShow = cv2.resize(imgOriginal, (widthX, widthY))
    im = Image.fromarray(imgToShow)

    myImage = ImageTk.PhotoImage(image=im)
    myImageLabel.config(image=myImage)
    myImageLabel.image = myImage

    # Print histogram
    makeAProbabilityPlot(predictionsSorted, sortedLabels)


# Create main window
root = Tk()
root.title("Sign recognition - deep learning model")
root.geometry('640x480')

openFileButton = Button(root, text="Open file", command=openImage)
imageLabel = Label(root)
classLabel = Label(root)
probabilityLabel = Label(root)
myImageLabel = Label(root)


openFileButton.pack()
imageLabel.pack()
classLabel.pack()
probabilityLabel.pack()
myImageLabel.pack()
root.mainloop()
