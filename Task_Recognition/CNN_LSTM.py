import os
import cv2
import sys
import numpy
import tensorflow
import tensorflow.keras
from tensorflow.keras.applications import MobileNet,MobileNetV2
from tensorflow.keras.applications import InceptionV3,ResNet50
from tensorflow.keras import layers
from tensorflow.keras.models import model_from_json

#tensorflow.compat.v1.enable_eager_execution()

class CNN_LSTM():
    def __init__(self):
        self.cnnModel = None
        self.lstmModel = None
        self.imageSequence = numpy.zeros((50,8))
        self.cnnLabels = None
        self.lstmLabels = None

    def loadModel(self,modelFolder):
        with open(os.path.join(modelFolder,"cnn_labels.txt"),"r") as f:
            self.cnnLabels = f.readlines()
        self.cnnLabels = numpy.array([x.replace("\n","") for x in self.cnnLabels])

        with open(os.path.join(modelFolder,"lstm_labels.txt"),"r") as f:
            self.lstmLabels = f.readlines()
        self.lstmLabels = numpy.array([x.replace("\n","") for x in self.lstmLabels])

        self.cnnModel = self.loadCNNModel(modelFolder)
        self.lstmModel = self.loadLSTMModel(modelFolder)




    def loadCNNModel(self,modelFolder):
        structureFileName = 'resnet50.json'
        weightsFileName = 'resnet50.h5'
        modelFolder = modelFolder.replace("'","")
        with open(os.path.join(modelFolder, structureFileName), "r") as modelStructureFile:
            JSONModel = modelStructureFile.read()
        #model = model_from_json(JSONModel)
        model = self.createCNNModel((224,224,3),num_classes=len(self.cnnLabels))
        model.load_weights(os.path.join(modelFolder, weightsFileName))
        return model

    def loadLSTMModel(self,modelFolder):
        structureFileName = 'single_LSTM.json'
        weightsFileName = 'single_LSTM.h5'
        modelFolder = modelFolder.replace("'", "")
        model = self.createSingleLSTMModel(self.imageSequence.shape[0],len(self.cnnLabels),len(self.lstmLabels))
        '''with open(os.path.join(modelFolder, structureFileName), "r") as modelStructureFile:
            JSONModel = modelStructureFile.read()
        model = model_from_json(JSONModel)'''
        model.load_weights(os.path.join(modelFolder, weightsFileName))
        adam = tensorflow.keras.optimizers.Adam(learning_rate=0.00001)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def predict(self,image):
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image, (224, 224)) #MobileNet
        #resized = cv2.resize(image, (299, 299))  #InceptionV3
        normImage = cv2.normalize(resized, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        normImage = numpy.expand_dims(normImage, axis=0)

        toolClassification = self.cnnModel.predict(numpy.array(normImage))
        toolIndex = numpy.argmax(toolClassification)
        toolLabel = self.cnnLabels[toolIndex]
        self.imageSequence = numpy.append(self.imageSequence[1:], toolClassification, axis=0)
        taskClassification = self.lstmModel.predict(numpy.array([self.imageSequence]))
        labelIndex = numpy.argmax(taskClassification)
        label = self.lstmLabels[labelIndex]
        networkOutput = str(label) + str(taskClassification)
        return networkOutput,toolLabel

    def createCNNModel(self,imageSize,num_classes):
        model = tensorflow.keras.models.Sequential()
        model.add(ResNet50(weights='imagenet',include_top=False,input_shape=imageSize,pooling='avg'))
        #model.add(InceptionV3(weights='imagenet', include_top=False, input_shape=imageSize))
        model.add(layers.Dense(512,activation='relu'))
        model.add(layers.Dense(num_classes,activation='softmax'))
        return model

    def createSingleLSTMModel(self,sequenceLength, cnnNumClasses,numOutputClasses):
        input = layers.Input(shape=(sequenceLength, cnnNumClasses))
        bLSTM = layers.Bidirectional(layers.LSTM(numOutputClasses, return_sequences=False))(input)
        r1 = layers.Dense(numOutputClasses, activation='relu')(bLSTM)
        r2 = layers.Dense(numOutputClasses, activation='relu')(r1)
        out = layers.Dense(numOutputClasses, activation='softmax')(r2)
        model = tensorflow.keras.models.Model(inputs=input, outputs=out)
        adam = tensorflow.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def saveModel(self,trainedCNNModel,trainedLSTMModel,saveLocation):
        JSONmodel = trainedCNNModel.to_json()
        structureFileName = 'resnet50.json'
        with open(os.path.join(saveLocation,structureFileName),"w") as modelStructureFile:
            modelStructureFile.write(JSONmodel)

        JSONmodel = trainedLSTMModel.to_json()
        structureFileName = 'single_LSTM.json'
        with open(os.path.join(saveLocation, structureFileName), "w") as modelStructureFile:
            modelStructureFile.write(JSONmodel)