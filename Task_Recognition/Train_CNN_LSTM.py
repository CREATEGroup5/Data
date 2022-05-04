import os
import sys
import numpy
import random
import pandas
import argparse
import tensorflow
import tensorflow.keras
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,LearningRateScheduler
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers
from tensorflow.keras.models import model_from_json
from tensorflow.keras import backend as K
import sklearn
import sklearn.model_selection
import sklearn.metrics
import cv2
import gc
from matplotlib import pyplot as plt
import CNN_LSTM
from CNNLSTMSequence import CNNSequence, LSTMSequence

#tensorflow.compat.v1.disable_eager_execution()

FLAGS = None

class Train_CNN_LSTM:

    def loadData(self,val_percentage,dataset):
        trainIndexes,valIndexes = sklearn.model_selection.train_test_split(dataset.index,test_size=val_percentage,shuffle=False)
        return trainIndexes,valIndexes

    def convertTextToNumericLabels(self,textLabels,labelValues):
        numericLabels =[]
        for i in range(len(textLabels)):
            label = numpy.zeros(len(labelValues))
            labelIndex = numpy.where(labelValues == textLabels[i])
            label[labelIndex] = 1
            numericLabels.append(label)
        return numpy.array(numericLabels)

    def saveTrainingInfo(self,saveLocation,trainingHistory,networkType,balanced=False):
        LinesToWrite = []
        modelType = "\nNetwork type: " + str(self.networkType)
        LinesToWrite.append(modelType)
        datacsv = "\nData CSV: " + str(FLAGS.data_csv_file)
        LinesToWrite.append(datacsv)
        numEpochs = "\nNumber of Epochs: " + str(len(trainingHistory["loss"]))
        numEpochsInt = len(trainingHistory["loss"])
        LinesToWrite.append(numEpochs)
        batch_size = "\nBatch size: " + str(self.batch_size)
        LinesToWrite.append(batch_size)
        if networkType == "LSTM":
            lstmSequenceLength = "\nSequence Length: " + str(self.sequenceLength)
            LinesToWrite.append(lstmSequenceLength)
            lstmSamplingRate = "\nDown sampling rate: " + str(self.downsampling)
            LinesToWrite.append(lstmSamplingRate)
            LearningRate = "\nLearning rate: " + str(self.lstm_learning_rate)
        else:
            LearningRate = "\nLearning rate: " + str(self.cnn_learning_rate)

        LinesToWrite.append(LearningRate)
        dataBalance = "\nData balanced: " + str(balanced)
        LinesToWrite.append(dataBalance)
        LossFunction = "\nLoss function: " + str(self.loss_Function)
        LinesToWrite.append(LossFunction)
        trainStatsHeader = "\n\nTraining Statistics: "
        LinesToWrite.append(trainStatsHeader)
        trainLoss = "\n\tFinal training loss: " + str(trainingHistory["loss"][numEpochsInt-1])
        LinesToWrite.append(trainLoss)
        for i in range(len(self.metrics)):
            trainMetrics = "\n\tFinal training " + self.metrics[i] + ": " + str(trainingHistory[self.metrics[i]][numEpochsInt-1])
            LinesToWrite.append(trainMetrics)
        valLoss = "\n\tFinal validation loss: " + str(trainingHistory["val_loss"][numEpochsInt - 1])
        LinesToWrite.append(valLoss)
        for i in range(len(self.metrics)):
            valMetrics = "\n\tFinal validation " + self.metrics[i] + ": " + str(trainingHistory["val_"+self.metrics[i]][numEpochsInt-1])
            LinesToWrite.append(valMetrics)

        with open(os.path.join(saveLocation,"trainingInfo_"+networkType+".txt"),'w') as f:
            f.writelines(LinesToWrite)

    def saveTrainingPlot(self,saveLocation,history,metric,networkType):
        fig = plt.figure()
        numEpochs =len(history[metric])
        plt.plot([x for x in range(numEpochs)], history[metric], 'bo', label='Training '+metric)
        plt.plot([x for x in range(numEpochs)], history["val_" + metric], 'b', label='Validation '+metric)
        plt.title(networkType+' Training and Validation ' + metric)
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.savefig(os.path.join(saveLocation, networkType+'_'+metric + '.png'))
        plt.close(fig)

    def splitVideoIntoSequences(self, images,sequenceLength=5, downsampling=2):
        number_of_sequences = len(images) - ((sequenceLength) * downsampling)
        sequences = []
        prevSequences = [[-1 for i in range(sequenceLength)] for j in range(downsampling)]
        for x in range(sequenceLength):
            for y in range(downsampling):
                newSeq = prevSequences[y][1:]
                newSeq.append(images[(x * downsampling) + y])
                prevSequences[y] = newSeq
                sequences.append(prevSequences[y])
        for x in range(0, number_of_sequences):
            sequences.append([images[i] for i in range(x, (x + sequenceLength * downsampling), downsampling)])
        return sequences

    def splitDatasetIntoSequences(self, indexes,sequenceLength=5, downsampling=2):
        entries = self.dataCSVFile.iloc[indexes]
        videoNames = entries["Folder"].unique()
        allSequences = []
        for video in videoNames:
            videoImages = entries.loc[entries["Folder"]==video]
            videoImageIndexes = videoImages.index
            videoSequences = self.splitVideoIntoSequences(videoImageIndexes,sequenceLength,downsampling)
            for sequence in videoSequences:
                allSequences.append(sequence)
        return allSequences



    def balanceDataset(self,dataset):
        videos = dataset["Folder"].unique()
        balancedFold = pandas.DataFrame(columns=dataset.columns)
        for vid in videos:
            images = dataset.loc[dataset["Folder"] == vid]
            labels = sorted(images["Tool"].unique())
            counts = images["Tool"].value_counts()
            print(vid)
            smallestCount = counts[counts.index[-1]]
            print("Smallest label: " + str(counts.index[-1]))
            print("Smallest count: " + str(smallestCount))
            if smallestCount == 0:
                print("Taking second smallest")
                secondSmallest = counts[counts.index[-2]]
                print("Second smallest count: " + str(secondSmallest))
                reducedLabels = [x for x in labels if x != counts.index[-1]]
                print(reducedLabels)
                for label in reducedLabels:
                    toolImages = images.loc[images["Tool"] == label]
                    randomSample = toolImages.sample(n=secondSmallest)
                    balancedFold = balancedFold.append(randomSample, ignore_index=True)
            else:
                for label in labels:
                    toolImages = images.loc[images["Tool"] == label]
                    if label == counts.index[-1]:
                        balancedFold = balancedFold.append(toolImages, ignore_index=True)
                    else:
                        randomSample = toolImages.sample(n=smallestCount)
                        balancedFold = balancedFold.append(randomSample, ignore_index=True)
        print(balancedFold["Tool"].value_counts())
        return balancedFold

    def createBalancedCNNDataset(self, trainSet, valSet):
        newCSV = pandas.DataFrame(columns=self.dataCSVFile.columns)
        resampledTrainSet = self.balanceDataset(trainSet)
        sortedTrain = resampledTrainSet.sort_values(by=['FileName'])
        sortedTrain["Set"] = ["Train" for i in sortedTrain.index]
        newCSV = newCSV.append(sortedTrain, ignore_index=True)
        resampledValSet = self.balanceDataset(valSet)
        sortedVal = resampledValSet.sort_values(by=['FileName'])
        sortedVal["Set"] = ["Validation" for i in sortedVal.index]
        newCSV = newCSV.append(sortedVal, ignore_index=True)
        print("Resampled Train Counts")
        print(resampledTrainSet["Tool"].value_counts())
        print("Resampled Validation Counts")
        print(resampledValSet["Tool"].value_counts())
        return newCSV

    def getBalancedSequences(self,sequences):
        sequenceLabels = self.getSequenceLabels(sequences)
        tempDataFrame = pandas.DataFrame({"Sequences":sequences,"Labels":sequenceLabels})
        balancedFold = pandas.DataFrame(columns = tempDataFrame.columns)
        counts = tempDataFrame["Labels"].value_counts()
        print("Initial Counts: ")
        print(counts)
        smallestCount = counts[counts.index[-1]]
        print("Smallest label: " + str(counts.index[-1]))
        print("Smallest count: " + str(smallestCount))
        if smallestCount == 0:
            print("Taking second smallest")
            secondSmallest = counts[counts.index[-2]]
            print("Second smallest count: " + str(secondSmallest))
            reducedLabels = [x for x in self.lstmLabelValues if x != counts.index[-1]]
            print(reducedLabels)
            for label in reducedLabels:
                taskSequences = tempDataFrame.loc[tempDataFrame["Labels"] == label]
                randomSample = taskSequences.sample(n=secondSmallest)
                balancedFold = balancedFold.append(randomSample, ignore_index=True)
        else:
            for label in self.lstmLabelValues:
                taskSequences = tempDataFrame.loc[tempDataFrame["Labels"] == label]
                if label == counts.index[-1]:
                    balancedFold = balancedFold.append(taskSequences, ignore_index=True)
                else:
                    randomSample = taskSequences.sample(n=smallestCount)
                    balancedFold = balancedFold.append(randomSample, ignore_index=True)
        balancedSequences = []
        for i in balancedFold.index:
            balancedSequences.append(balancedFold["Sequences"][i])
        print("Resampled Sequence Counts")
        print(balancedFold["Labels"].value_counts())
        return balancedSequences

    def getSequenceLabels(self, sequences):
        sequenceLabels = []
        for sequence in sequences:
            textLabel = self.dataCSVFile["Overall Task"][sequence[len(sequence) - 1]]
            sequenceLabels.append(textLabel)
        return sequenceLabels

    def readImages(self, files,cnnModel):
        images = []
        numLoaded = 0
        allOutputs = numpy.array([])
        for i in range(len(files)):
            image = cv2.imread(files[i])
            resized_image = cv2.resize(image, (224, 224))
            normImg = cv2.normalize(resized_image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            images.append(normImg)
            numLoaded += 1
            if numLoaded % 500 == 0 or i == (len(files) - 1):
                print("loaded " + str(numLoaded) + ' / ' + str(len(files)) + ' images')
                if allOutputs.size == 0:
                    allOutputs = cnnModel.predict(numpy.array(images))
                else:
                    cnnOutput = cnnModel.predict(numpy.array(images))
                    allOutputs = numpy.append(allOutputs, cnnOutput, axis=0)
                del images
                gc.collect()
                images = []
            del image
            del resized_image
            del normImg
        return allOutputs

    def train(self):
        if FLAGS.save_location == "":
            print("No save location specified. Please set flag --save_location")
        elif FLAGS.data_csv_file == "":
            print("No dataset specified. Please set flag --data_csv_file")
        else:
            self.saveLocation = FLAGS.save_location
            self.networkType = "CNN_LSTM"
            self.dataCSVFile = pandas.read_csv(FLAGS.data_csv_file)
            self.validation_percentage = FLAGS.validation_percentage

            self.numEpochs = FLAGS.num_epochs_cnn
            self.numLSTMEpochs = FLAGS.num_epochs_lstm
            self.batch_size = FLAGS.batch_size
            self.sequenceLength = FLAGS.sequence_length
            self.downsampling = FLAGS.downsampling_rate
            self.cnn_learning_rate = FLAGS.cnn_learning_rate
            self.lstm_learning_rate = FLAGS.lstm_learning_rate
            self.balanceCNN = FLAGS.balance_CNN_Data
            self.balanceLSTM = FLAGS.balance_LSTM_Data
            self.cnn_optimizer = tensorflow.keras.optimizers.Adam(learning_rate=self.cnn_learning_rate)
            self.lstm_optimizer = tensorflow.keras.optimizers.Adam(learning_rate=self.lstm_learning_rate)
            self.loss_Function = self.focal_loss #FLAGS.loss_function
            self.metrics = FLAGS.metrics.split(",")
            network = CNN_LSTM.CNN_LSTM()
            if not os.path.exists(self.saveLocation):
                os.mkdir(self.saveLocation)
            taskLabelName = "Overall Task"
            toolLabelName = "Tool"

            TrainIndexes, ValIndexes = self.loadData(self.validation_percentage, self.dataCSVFile)
            if self.balanceCNN:
                trainData = self.dataCSVFile.iloc[TrainIndexes]
                valData = self.dataCSVFile.iloc[ValIndexes]
                self.dataCSVFile = self.createBalancedCNNDataset(trainData, valData)
                balancedTrainData = self.dataCSVFile.loc[self.dataCSVFile["Set"] == "Train"]
                balancedValData = self.dataCSVFile.loc[self.dataCSVFile["Set"] == "Validation"]
                TrainIndexes = balancedTrainData.index
                ValIndexes = balancedValData.index

            cnnTrainDataSet = CNNSequence(self.dataCSVFile, TrainIndexes, self.batch_size, toolLabelName, augmentations=True)
            cnnValDataSet = CNNSequence(self.dataCSVFile, ValIndexes, self.batch_size, toolLabelName)

            cnnLabelValues = numpy.array(sorted(self.dataCSVFile[toolLabelName].unique()))
            numpy.savetxt(os.path.join(self.saveLocation,"cnn_labels.txt"),cnnLabelValues,fmt='%s',delimiter=',')

            cnnModel = network.createCNNModel((224,224,3),num_classes=len(cnnLabelValues))
            cnnModel.compile(optimizer = self.cnn_optimizer, loss = self.loss_Function, metrics = self.metrics)

            earlyStoppingCallback = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
            modelCheckPointCallback = ModelCheckpoint(os.path.join(self.saveLocation,'resnet50.h5'), verbose=1,monitor='val_accuracy', mode='max', save_weights_only = True,save_best_only=True)

            history = cnnModel.fit(x=cnnTrainDataSet,
                                   validation_data=cnnValDataSet,
                                   epochs=self.numEpochs,callbacks=[modelCheckPointCallback,earlyStoppingCallback])

            cnnModel.load_weights(os.path.join(self.saveLocation, 'resnet50.h5'))

            self.saveTrainingInfo(self.saveLocation, history.history,"CNN")
            self.saveTrainingPlot(self.saveLocation, history.history, "loss", "CNN")
            for metric in self.metrics:
                self.saveTrainingPlot(self.saveLocation,history.history,metric,"CNN")

            allSequences = self.splitDatasetIntoSequences(self.dataCSVFile.index, sequenceLength=self.sequenceLength,downsampling=self.downsampling)
            lstmTrainSequences,lstmValSequences = sklearn.model_selection.train_test_split(allSequences,test_size=self.validation_percentage)

            if self.balanceLSTM:
                lstmTrainSequences = self.getBalancedSequences(lstmTrainSequences)
                lstmValSequences = self.getBalancedSequences(lstmValSequences)

            self.lstmLabelValues = numpy.array(sorted(self.dataCSVFile[taskLabelName].unique()))
            numpy.savetxt(os.path.join(self.saveLocation, "lstm_labels.txt"), self.lstmLabelValues, fmt='%s', delimiter=',')

            inputs = self.readImages([os.path.join(self.dataCSVFile["Folder"][x], self.dataCSVFile["FileName"][x]) for x in self.dataCSVFile.index],cnnModel)
            lstmTrainDataSet = LSTMSequence(self.dataCSVFile, inputs, lstmTrainSequences, cnnModel, self.batch_size, taskLabelName)
            print("Training images loaded")
            lstmValDataSet = LSTMSequence(self.dataCSVFile, inputs, lstmValSequences, cnnModel, self.batch_size, taskLabelName)
            print("Validation images loaded")

            modelCheckPointCallback = ModelCheckpoint(os.path.join(self.saveLocation, 'single_LSTM.h5'), verbose=1,
                                                      monitor='val_accuracy', mode='max', save_weights_only=True,
                                                      save_best_only=True)
            earlyStoppingCallback = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

            lstmModel = network.createSingleLSTMModel(self.sequenceLength, cnnNumClasses = len(cnnLabelValues),numOutputClasses=len(self.lstmLabelValues))
            lstmModel.compile(optimizer=self.lstm_optimizer, loss=self.loss_Function, metrics=self.metrics,run_eagerly=False)

            history = lstmModel.fit(x=lstmTrainDataSet,
                                   validation_data=lstmValDataSet,
                                   epochs=self.numLSTMEpochs,callbacks=[modelCheckPointCallback,earlyStoppingCallback])
            lstmModel.load_weights(os.path.join(self.saveLocation, 'single_LSTM.h5'))

            self.saveTrainingInfo(self.saveLocation, history.history, "LSTM")
            self.saveTrainingPlot(self.saveLocation, history.history, "loss","LSTM")
            for metric in self.metrics:
                self.saveTrainingPlot(self.saveLocation, history.history, metric, "LSTM")

            network.saveModel(cnnModel,lstmModel,self.saveLocation)
            
    def focal_loss_with_logits(self, logits, targets, alpha, gamma, y_pred):
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)

        return (tensorflow.math.log1p(tensorflow.exp(-tensorflow.abs(logits))) + tensorflow.nn.relu(
            -logits)) * (weight_a + weight_b) + logits * weight_b

    def focal_loss(self, y_true, y_pred):
        y_pred = tensorflow.clip_by_value(y_pred, tensorflow.keras.backend.epsilon(),
                                  1 - tensorflow.keras.backend.epsilon())
        logits = tensorflow.math.log(y_pred / (1 - y_pred))

        loss = self.focal_loss_with_logits(logits=logits, targets=y_true,
                                      alpha=0.25, gamma=2, y_pred=y_pred)

        return tensorflow.reduce_mean(loss)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--save_location',
      type=str,
      default='',
      help='Name of the directory where the models and results will be saved'
  )
  parser.add_argument(
      '--validation_percentage',
      type=float,
      default=0.3,
      help='percent of data to be used for validation'
  )
  parser.add_argument(
      '--data_csv_file',
      type=str,
      default='',
      help='Path to the csv file containing locations for all data used in training'
  )
  parser.add_argument(
      '--num_epochs_cnn',
      type=int,
      default=50,
      help='number of epochs used in training the cnn'
  )
  parser.add_argument(
      '--num_epochs_lstm',
      type=int,
      default=8,
      help='number of epochs used in training the lstm'
  )
  parser.add_argument(
      '--sequence_length',
      type=int,
      default=50,
      help='number of images in the sequences for task prediction'
  )
  parser.add_argument(
      '--downsampling_rate',
      type=int,
      default=4,
      help='number of images to skip while creating training sequences'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=8,
      help='number of images in each batch'
  )
  parser.add_argument(
      '--cnn_learning_rate',
      type=float,
      default=0.00001,
      help='Learning rate used in training cnn network'
  )
  parser.add_argument(
      '--lstm_learning_rate',
      type=float,
      default=0.0001,
      help='Learning rate used in training lstm network'
  )
  parser.add_argument(
      '--balance_CNN_Data',
      type=bool,
      default=False,
      help='Whether or not to balance the data by class when training the CNN'
  )
  parser.add_argument(
      '--balance_LSTM_Data',
      type=bool,
      default=False,
      help='Whether or not to balance the data by class when training the LSTM'
  )
  parser.add_argument(
      '--loss_function',
      type=str,
      default='categorical_crossentropy',
      help='Name of the loss function to be used in training (see keras documentation).'
  )
  parser.add_argument(
      '--metrics',
      type=str,
      default='accuracy',
      help='Metrics used to evaluate model.'
  )

FLAGS, unparsed = parser.parse_known_args()
tm = Train_CNN_LSTM()
tm.train()
