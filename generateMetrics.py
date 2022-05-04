import pandas as pd
import sklearn.metrics

def classWeightedAccuracy(csv1, csv2):
    groundTruth = pd.read_csv(csv1)
    predictions = pd.read_csv(csv2)
    tasksTruth = groundTruth["Overall Task"]



    pass

def averagePrecision():
    groundTruth = pd.read_csv(csv1)
    predictions = pd.read_csv(csv2)
    tasksTruth = groundTruth["Overall Task"]
    toolsTruth = groundTruth["Tool"]
    predTruth = predictions["Overall Task"]
    toolsPred = predictions["Tool"]

    precision1 = sklearn.metrics.average_precision_score(tasksTruth, tasksPred, average='macro', pos_label=1, sample_weight=None)
    precision2 = sklearn.metrics.average_precision_score(toolsTruth, toolsPred, average='macro', pos_label=1, sample_weight=None)

    print("Precision for tasks: ", precision1)
    print("Precision for tools: ", precision2)

def averageRecall():
    pass

def averageTransitionalDelay(csv1, csv2):
    columns = ["FileName", "Time Recorded", "Overall Task"]
    groundTruth = pd.read_csv(csv1)
    predictions = pd.read_csv(csv2)
    videos = groundTruth["Folder"].unique()
    tasks = sorted(groundTruth["Overall Task"].unique())
    tasks.remove("nothing")
    startAverages = []
    endAverages = []

    for task in tasks:
        firstTruthTotal = 0
        lastTruthTotal = 0
        firstPredTotal = 0
        lastPredTotal = 0

        for vid in videos:
            # Extract video rows
            imagesTruth = groundTruth.loc[groundTruth["Folder"] == vid]
            imagesPredict = predictions.loc[predictions["Folder"] == vid]

            # Extract task instances from video
            taskOccurrences = imagesTruth.loc[imagesTruth["Overall Task"] == task]
            if not taskOccurrences.empty:

                # check the head and tail for when the task starts and stops
                firstOccurrence = taskOccurrences["Time Recorded"][taskOccurrences.index[0]]
                lastOccurrence = taskOccurrences["Time Recorded"][taskOccurrences.index[-1]]

                firstTruthTotal += firstOccurrence
                lastTruthTotal += lastOccurrence

            # Extract task instances from video
            taskOccurrences2 = imagesPredict.loc[imagesPredict["Overall Task"] == task]
            if not taskOccurrences2.empty:
            # check the head and tail for when the task starts and stops
                firstOccurrence2 = taskOccurrences2["Time Recorded"][taskOccurrences2.index[0]]
                lastOccurrence2 = taskOccurrences2["Time Recorded"][taskOccurrences2.index[-1]]

                firstPredTotal += firstOccurrence2
                lastPredTotal += lastOccurrence2

        firstDiff = abs((firstTruthTotal / len(videos)) - firstPredTotal / len(videos))
        lastDiff = abs((lastTruthTotal / len(videos)) - lastPredTotal / len(videos))

        print("Start absolute difference for task" + task + ": " + str(firstDiff))
        print("End absolute difference for task" + task + ": " + str(lastDiff))
        startAverages.append(firstDiff)
        endAverages.append(lastDiff)
    print()
    print("Average start over all tasks: " + str(sum(startAverages)/len(startAverages)))
    print("Average end over all tasks: " + str(sum(endAverages)/len(endAverages)))


if __name__ == '__main__':
    averageTransitionalDelay("C:\\Users\\perklab\\Documents\\CREATE_Challenge\\Training_Data\\AN01-20210104-154854\\AN01-20210104-154854_Labels.csv",
                             "C:\\Users\\perklab\\Documents\\taskDetectionRun6\\Task_predictions.csv")







