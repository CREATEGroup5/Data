import os
import cv2
import sys
import numpy
import pandas
import tensorflow
import tensorflow.keras
from tensorflow.keras.applications import MobileNet,MobileNetV2
from tensorflow.keras.applications import InceptionV3,ResNet50
from tensorflow.keras import layers
from tensorflow.keras.models import model_from_json
import CNN_LSTM
import argparse

FLAGS = None

class Predict_CNN_LSTM:
    def getPredictions(self):
        network = CNN_LSTM.CNN_LSTM()
        if FLAGS.save_location == "":
            print("No save location specified. Please set flag --save_location")
        elif FLAGS.data_csv_file == "":
            print("No dataset specified. Please set flag --data_csv_file")
        else:
            self.saveLocation = FLAGS.save_location
            self.networkType = "CNN_LSTM"
            self.dataCSVFile = pandas.read_csv(FLAGS.data_csv_file)
            self.sequenceLength = FLAGS.sequence_length

            network.loadModel(self.saveLocation)
            numClasses = len(network.cnnLabels)
            print(numClasses)
            network.imageSequence = numpy.zeros((self.sequenceLength,numClasses))
            nothingIndex = numpy.where(network.cnnLabels == "nothing")
            network.imageSequence[:,nothingIndex[0][0]] = numpy.ones((self.sequenceLength,))
            columns = ["FileName", "Time Recorded", "Tool", "Overall Task"] + [i for i in network.lstmLabels]
            predictions = pandas.DataFrame(columns=columns)
            predictions["FileName"] = self.dataCSVFile["FileName"]
            predictions["Time Recorded"] = self.dataCSVFile["Time Recorded"]
            initialFolder = self.dataCSVFile["Folder"][0]
            for i in self.dataCSVFile.index:
                if i%500 == 0 or i==len(self.dataCSVFile.index)-1:
                    print("{}/{} predictions generated".format(i,len(self.dataCSVFile.index)))
                if self.dataCSVFile["Folder"][i] != initialFolder:
                    network.imageSequence = numpy.zeros((self.sequenceLength, numClasses))
                    nothingIndex = numpy.where(network.cnnLabels == "nothing")
                    print(nothingIndex[0][0])
                    network.imageSequence[:, nothingIndex[0][0]] = numpy.ones((self.sequenceLength,))
                    initialFolder = self.dataCSVFile["Folder"][i]
                image = cv2.imread(os.path.join(self.dataCSVFile["Folder"][i],self.dataCSVFile["FileName"][i]))
                taskPrediction,toolLabel = network.predict(image)
                taskLabel,confidences = taskPrediction.split('[[')
                predictions["Overall Task"][i] = taskLabel
                predictions["Tool"][i] = toolLabel
            predictions.to_csv(os.path.join(self.saveLocation,"Task_Predictions.csv"),index=False)
            print("Predictions saved to: {}".format(os.path.join(self.saveLocation,"Task_Predictions.csv")))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--save_location',
      type=str,
      default='',
      help='Name of the directory where the saved model is located'
  )
  parser.add_argument(
      '--data_csv_file',
      type=str,
      default='',
      help='Path to the csv file containing locations for all data used in testing'
  )
  parser.add_argument(
      '--sequence_length',
      type=int,
      default=50,
      help='number of images in the sequences for task prediction'
  )

FLAGS, unparsed = parser.parse_known_args()
tm = Predict_CNN_LSTM()
tm.getPredictions()

