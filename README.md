<p align="center"><img width=80% src="https://github.com/Fishu99/SignRecognition/blob/master/Media/SignRecognition.png"></p>

# Road sign recognition project 🪧

Program uses a dataset of traffic signs pictures for learning how to classify the traffic signs and how to recognize them on image.<br>
Dataset of road signs is a dataset from Kaggle website which is available under <a href="https://www.kaggle.com/datasets/flo2607/traffic-signs-classification">this</a> link.

## 📋 About app
 The program presents a histogram of recognized traffic signs probability and presents the most accurate prediction after it processes the given picture. Project is split into four main parts.
 
First one's task is to split the original dataset into three different groups - training, test and validation sets. Data unification helps to determine how much influence the previously set parameters have on the trained models. Below are presented initial settings for the **_directories.py_** 

```python
imagesPath = "OriginalData"  # Catalog with data
storedTrainPath = "TrainData"   # Catalog with data for training
storedTestPath = "TestData"     # Catalog with data for testing
storedValidationPath = "ValidationData"     # Catalog with data for validating
labelsCsv = "labels.csv"  # File with sign labels
imgDims = (32, 32, 3)   # Training Image dimensions
testRatio = 0.2     # It represents the proportion of the dataset to include in the test split
validationRatio = 0.2   # It represents the proportion of the dataset to include in the validation split 
```

Second one is responsible for creating and training models with given parameters like number of epochs and steps of training in each of them. Images from those sets are being processed and the training set is also being augmented before model creation. Mentioned parameters for **_main.py_** are presented below:

```python
storedTrainPath = "TrainData"   # Catalog with data for training
storedTestPath = "TestData"     # Catalog with data for testing
storedValidationPath = "ValidationData"     # Catalog with data for validating
labelsCsv = "labels.csv"  # File with sign labels
modelOutput = "Model_03" # Name of the generated model
imgDims = (32, 32, 3)   # Training Image dimensions
epochsCount = 20 # Amount of epochs
stepsPerEpoch = 250     # Number of steps in each epoch of training - should be trainX/batchSize
batchSize = 50  # Number of elements to process together
```

During training histogram of read data per road sign class is shown to the user for a better view:

<p align="center"><img width=80% src="https://github.com/Fishu99/SignRecognition/blob/master/Media/sampleDistribution.png"></p>

After completing training a model it is being tested by using the previously created _Test_ directory of images. Results of the test are presented in the console and by generated charts:

<p align="center"><img width=80% src="https://github.com/Fishu99/SignRecognition/blob/master/Media/Media/modelResult.png"></p>

Third part produces a confusion matrix from the selected model to make sure of its correctness. Confusion Matrix makes it possible to obtain information about which signs the model trained is confused with. This is very useful information, meaning that more pictures with this road sign should be added to the dataset for better results and better accuracy of the model. 

<p align="center"><img width=80% src="https://github.com/Fishu99/SignRecognition/blob/master/Media/cmatrix.png"></p>

Last part stands for the Graphical User Interface application. It uses previously trained model to recognize traffic signs on selected image and produce a histogram of classes, where we can see the probability of choosing this particular traffic sign.

<p align="center"><img width=80% src="https://github.com/Fishu99/SignRecognition/blob/master/Media/usage.png"></p>

## 🔐 Requirements
To run your app, the user will need to have the following installed:
- Python (Python 3.x)
- Required Python packages: 
  - NumPy - for array operations
  - OpenCV - image processing
  - Matplotlib - creating plots
  - TensorFlow - actions on model
  - Tkinter - graphical user interface
  - Keras - model operations
  - Pandas - reading from .csv file
  - Sklearn - splitting data to train, test and validation arrays
- Data images files that are available under <a href="https://www.kaggle.com/datasets/flo2607/traffic-signs-classification">kaggle dataset</a>.