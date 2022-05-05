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
import Faster_RCNN
import argparse

FLAGS = None

class Predict_Faster_RCNN:
    def getPredictions(self):
        network = Faster_RCNN.Faster_RCNN()
        if FLAGS.save_location == "":
            print("No save location specified. Please set flag --save_location")
        elif FLAGS.data_csv_file == "":
            print("No dataset specified. Please set flag --data_csv_file")
        else:
            self.saveLocation = FLAGS.save_location
            self.dataCSVFile = pandas.read_csv(FLAGS.data_csv_file)

            network.loadModel(self.saveLocation)

            columns = ["FileName", "Time Recorded", "Tool bounding box"]
            predictions = pandas.DataFrame(columns=columns)
            predictions["FileName"] = self.dataCSVFile["FileName"]
            predictions["Time Recorded"] = self.dataCSVFile["Time Recorded"]
            for i in self.dataCSVFile.index:
                if i%10 == 0 or i==len(self.dataCSVFile.index)-1:
                    print("{}/{} predictions generated".format(i,len(self.dataCSVFile.index)))
                image = cv2.imread(os.path.join(self.dataCSVFile["Folder"][i],self.dataCSVFile["FileName"][i]))
                toolBoundingBoxes = network.predict(image)
                predictions["Tool bounding box"][i] = toolBoundingBoxes
            predictions.to_csv(os.path.join(self.saveLocation,"BoundingBox_Predictions.csv"),index=False)
            print("Predictions saved to: {}".format(os.path.join(self.saveLocation,"BoundingBox_Predictions.csv")))


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

FLAGS, unparsed = parser.parse_known_args()
tm = Predict_Faster_RCNN()
tm.getPredictions()

