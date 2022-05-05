import pandas
import os
import argparse
import shutil

FLAGS = None

def unpackZipFiles(dataLocation,datasetType):
    if datasetType == "Train":
        for i in range(1, 5):
            print("Extracting data from: Training_Data_Part{}.zip".format(i))
            extractionDir = os.path.join(dataLocation, "Training_Data_Part{}".format(i))
            if not os.path.exists(extractionDir):
                os.mkdir(extractionDir)
            zipFile = os.path.join(dataLocation, "Training_Data_Part{}.zip".format(i))
            shutil.unpack_archive(zipFile,extractionDir)
    else:
        print("Extracting data from: Test_Data.zip")
        extractionDir = os.path.join(dataLocation, "Test_Data")
        if not os.path.exists(extractionDir):
            os.mkdir(extractionDir)
        zipFile = os.path.join(dataLocation, "Test_Data.zip")
        shutil.unpack_archive(zipFile, extractionDir)

def moveFilesToMainDirectory(videoDir,mainDirectory):
    videoID = os.path.basename(videoDir)
    print("Transferring data from video: {}".format(videoID))
    newlocation = os.path.join(mainDirectory,videoID)
    if not os.path.exists(newlocation):
        os.mkdir(newlocation)
    fileNames = os.listdir(videoDir)
    for file in fileNames:
        oldFileLocation = os.path.join(videoDir,file)
        newFileLocation = os.path.join(newlocation,file)
        shutil.move(oldFileLocation,newFileLocation)

def createMainDatasetCSV(mainDirectory,datasetType):
    print("Creating main dataset csv")

    if datasetType == "Train":
        dataCSVFileName = "Training_Data.csv"
    else:
        dataCSVFileName = "Test_Data.csv"
    videoIDs = [x for x in os.listdir(mainDirectory) if not "." in x]
    i = 0
    dataCSVInitialized = False
    for video in videoIDs:
        labelFile = pandas.read_csv(os.path.join(mainDirectory,video,video+"_Labels.csv"))
        labelFile["Folder"] = [os.path.join(mainDirectory,video) for i in range(len(labelFile.index))]
        if not dataCSVInitialized:
            dataCSV = pandas.DataFrame(columns=labelFile.columns)
            dataCSVInitialized = True
        dataCSV = dataCSV.append(labelFile,ignore_index=True)
    for column in dataCSV.columns:
        if "Unnamed" in column:
            dataCSV = dataCSV.drop(column,axis=1)
    dataCSV.to_csv(os.path.join(mainDirectory,dataCSVFileName),index = False)

def createDataset():
    baseLocation = FLAGS.compressed_location
    targetLocation = FLAGS.target_location
    datasetType = FLAGS.dataset_type
    unpackZipFiles(baseLocation,datasetType)
    if datasetType == "Train":
        dataSetLocation = os.path.join(targetLocation,"Training_Data")
        if not os.path.exists(dataSetLocation):
            os.mkdir(dataSetLocation)
        for i in range(1, 5):
            dataFolder = os.path.join(baseLocation, "Training_Data_Part{}".format(i))
            for videoDir in os.listdir(dataFolder):
                moveFilesToMainDirectory(os.path.join(dataFolder, videoDir), dataSetLocation)
            shutil.rmtree(dataFolder)
            print("Removed empty directory {}".format(dataFolder))
    else:
        dataSetLocation = os.path.join(targetLocation,"Test_Data")
        if not os.path.exists(dataSetLocation):
            os.mkdir(dataSetLocation)
        dataFolder = os.path.join(baseLocation, "Test_Data")
        for videoDir in os.listdir(dataFolder):
            moveFilesToMainDirectory(os.path.join(dataFolder, videoDir), dataSetLocation)
        shutil.rmtree(dataFolder)
        print("Removed empty directory {}".format(dataFolder))
    createMainDatasetCSV(dataSetLocation,datasetType)
    print("Dataset preparation complete. Data located in directory: {}".format(dataSetLocation))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--compressed_location',
      type=str,
      default='',
      help='Name of the directory where the compressed data files are located'
  )
  parser.add_argument(
      '--target_location',
      type=str,
      default='',
      help='Name of the directory where the uncompressed data files will be located'
  )
  parser.add_argument(
      '--dataset_type',
      type=str,
      default='Train',
      help='Type of Dataset you are creating: should be Train or Test'
  )

  FLAGS, unparsed = parser.parse_known_args()
  createDataset()