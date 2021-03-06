from __future__ import division
import os
import gc
import numpy as np
import pandas
import argparse
import math
import random
import time
import pickle
import cv2
import sklearn
import sklearn.model_selection
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from keras_frcnn import config, data_generators
from keras_frcnn import losses as lossFunctions
import keras_frcnn.roi_helpers as roi_helpers
from tensorflow.keras import utils
from matplotlib import pyplot as plt
from keras_frcnn import resnet as nn
import Faster_RCNN

FLAGS = None

class Train_Faster_RCNN:

    #Loads the data from the specified CSV file
    # fold: The fold number given in the CSV file (should be an int)
    # set: which set the images and labels make up (should be one of: "Train","Validation", or "Test")
    # Returns:
    #   images: The list of images loaded from the files or girderIDs
    #   imageLabels: a dictionary of the labels for each image, indexed by the label name
    def loadData(self, val_percentage, dataset):
        baseFolder = os.path.basename(dataset["Folder"][0])
        faultyFolder = os.path.join(baseFolder,"AN01-20210104-154854")
        dataset = dataset.loc[dataset["Folder"] != faultyFolder]
        dataset.index = [i for i in range(len(dataset.index))]
        trainIndexes, valIndexes = sklearn.model_selection.train_test_split(dataset.index, test_size=val_percentage,
                                                                            shuffle=False)
        trainData = dataset.iloc[trainIndexes]
        trainData.index = [i for i in range(len(trainIndexes))]
        valData = dataset.iloc[valIndexes]
        valData.index = [i for i in range(len(valIndexes))]
        return trainData, valData

    def getClassMapping(self,dataCSV,labelName):
        class_names = []
        class_indexes = {}
        classes_count = {}
        numProcessed = 0
        numToProcess = len(dataCSV.index)
        for i in dataCSV.index:
            if numProcessed % 1000 == 0:
                print("Processed rows {} / {}".format(numProcessed,numToProcess))
            numProcessed +=1
            strBoundBox = str(dataCSV[labelName][i])
            if strBoundBox =="[]":
                dataCSV = dataCSV.drop(i)
            else:
                if self.selectedLabels != None:
                    for label in self.selectedLabels:
                        if label in strBoundBox:
                            inSelectedLabels = True
                        else:
                            inSelectedLabels = False
                else:
                    inSelectedLabels = True
                if inSelectedLabels:
                    strBoundBox = strBoundBox.replace(" ","")
                    strBoundBox = strBoundBox.replace("'", "")
                    strBoundBox = strBoundBox.replace("[", "")
                    strBoundBox = strBoundBox.replace("]", "")
                    boundingBoxes = []
                    if strBoundBox != "":
                        listBoundBox = strBoundBox.split("},{")
                        for boundingBox in listBoundBox:
                            boundingBox = boundingBox.replace("{", "")
                            boundingBox = boundingBox.replace("}", "")
                            keyEntryPairs = boundingBox.split(",")
                            boundingBoxDict = {}
                            for pair in keyEntryPairs:
                                key, entry = pair.split(":")
                                if entry.isnumeric():
                                    boundingBoxDict[key] = int(entry)
                                else:
                                    boundingBoxDict[key] = entry
                            x1 = boundingBoxDict["xmin"]
                            x2 = boundingBoxDict["xmax"]
                            y1 = boundingBoxDict["ymin"]
                            y2 = boundingBoxDict["ymax"]
                            try:
                                boundingBoxDict["xmin"] = min(x1,x2)
                                boundingBoxDict["xmax"] = max(x1,x2)
                                boundingBoxDict["ymin"] = min(y1,y2)
                                boundingBoxDict["ymax"] = max(y1,y2)
                                if self.selectedLabels == None or boundingBoxDict['class'] in self.selectedLabels:
                                    boundingBoxes.append(boundingBoxDict)
                                    if not boundingBoxDict['class'] in class_names:
                                        class_names.append(boundingBoxDict['class'])
                                        classes_count[boundingBoxDict['class']] = 1
                                    else:
                                        classes_count[boundingBoxDict['class']] += 1
                                    try:
                                        class_indexes[boundingBoxDict['class']].append(i)
                                    except KeyError:
                                        class_indexes[boundingBoxDict['class']] = [i]
                            except TypeError:
                                print("Incompatible types for bounding box: {}".format(boundingBoxDict))
                    dataCSV[labelName][i] = boundingBoxes
                else:
                    dataCSV = dataCSV.drop(i)
        if self.balanceDataset:
            dataCSV,classes_count = self.balanceSamples(dataCSV,class_indexes)
        imageMeans = self.getImageMeans(dataCSV)
        dataCSV.index = [i for i in range(len(dataCSV.index))]
        numericClassNames = [x for x in range(len(class_names))]
        class_mapping = dict(zip(class_names, numericClassNames))
        dataCSV["cropped"] = [False for x in dataCSV.index]
        return classes_count, class_mapping, dataCSV,imageMeans

    def getImageMeans(self,dataCSV):
        imgChannelMeans = np.zeros((3,))
        for i in dataCSV.index:
            img = cv2.imread(os.path.join(dataCSV["Folder"][i],dataCSV["FileName"][i]))
            imgChannelMeans += np.mean(np.mean(img,axis=0),axis=0)
            del(img)
        gc.collect()
        imgChannelMeans = imgChannelMeans/len(dataCSV.index)
        return imgChannelMeans

    def balanceSamples(self,dataCSV,class_indexes,minCount = math.inf):
        newDataCSV = pandas.DataFrame(columns=dataCSV.columns)
        newClass_count = {}
        if self.selectedLabels !=None:
            for label in self.selectedLabels:
                count = len(class_indexes[label])
                if count < minCount:
                    minCount = count
            for label in self.selectedLabels:
                newClass_count[label] = minCount
                all_samples = dataCSV.loc[class_indexes[label],:]
                if len(all_samples.index) == minCount:
                    newDataCSV = newDataCSV.append(all_samples,ignore_index=True)
                else:
                    selectedSamples = all_samples.sample(n=minCount)
                    newDataCSV = newDataCSV.append(selectedSamples,ignore_index = True)
        else:
            for label in class_indexes:
                count = len(class_indexes[label])
                if count < minCount:
                    minCount = count
            for label in class_indexes:
                newClass_count[label] = minCount
                all_samples = dataCSV.loc[class_indexes[label],:]
                if len(all_samples.index) == minCount:
                    newDataCSV = newDataCSV.append(all_samples,ignore_index=True)
                else:
                    selectedSamples = all_samples.sample(n=minCount)
                    newDataCSV = newDataCSV.append(selectedSamples,ignore_index = True)
        return newDataCSV, newClass_count

    def convertTextToNumericLabels(self,textLabels,labelValues):
        numericLabels =[]
        for i in range(len(textLabels)):
            label = np.zeros(len(labelValues))
            labelIndex = np.where(labelValues == textLabels[i])
            label[labelIndex] = 1
            numericLabels.append(label)
        return np.array(numericLabels)

    def saveTrainingInfo(self,saveLocation,trainingHistory):
        LinesToWrite = []
        modelType = "\nNetwork type: " + str(self.networkType)
        LinesToWrite.append(modelType)
        datacsv = "\nData CSV: " + str(FLAGS.data_csv_file)
        LinesToWrite.append(datacsv)
        numEpochs = "\nNumber of Epochs: " + str(self.numEpochs)
        LinesToWrite.append(numEpochs)
        batch_size = "\nBatch size: " + str(self.batch_size)
        LinesToWrite.append(batch_size)
        LearningRate = "\nLearning rate: " + str(self.learning_rate)
        LinesToWrite.append(LearningRate)
        LossFunction = "\nLoss function: " + str(self.loss_Function)
        LinesToWrite.append(LossFunction)
        trainStatsHeader = "\n\nTraining Statistics: "
        LinesToWrite.append(trainStatsHeader)
        trainLoss = "\n\tFinal training loss: " + str(trainingHistory["loss"][len(trainingHistory)-1])
        LinesToWrite.append(trainLoss)
        for i in range(len(self.metrics)):
            trainMetrics = "\n\tFinal training " + self.metrics[i] + ": " + str(trainingHistory[self.metrics[i]][len(trainingHistory)-1])
            LinesToWrite.append(trainMetrics)
        trainLoss = "\n\tFinal validation loss: " + str(trainingHistory["val_loss"][len(trainingHistory) - 1])
        LinesToWrite.append(trainLoss)
        for i in range(len(self.metrics)):
            valMetrics = "\n\tFinal validation " + self.metrics[i] + ": " + str(trainingHistory["val_"+self.metrics[i]][len(trainingHistory) - 1])
            LinesToWrite.append(valMetrics)

        with open(os.path.join(saveLocation,"trainingInfo.txt"),'w') as f:
            f.writelines(LinesToWrite)

    def saveTrainingPlot(self,saveLocation,history,metric):
        fig = plt.figure()
        plt.plot([x for x in range(self.numEpochs)], history[metric], 'bo', label='Training '+metric)
        plt.plot([x for x in range(self.numEpochs)], history["val_" + metric], 'b', label='Validation '+metric)
        plt.title('Training and Validation ' + metric)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(saveLocation, metric + '.png'))

    def trainOneEpoch(self,epoch_num, epoch_length, dataGenerator, network, rpn_accuracy_rpn_monitor,labelName):
        progbar = utils.Progbar(epoch_length)
        print('Epoch {}/{}'.format(epoch_num, self.numEpochs))
        losses = np.zeros((epoch_length, 5))

        iter_num = 0
        rpn_accuracy_for_epoch = []
        start_time = time.time()

        while iter_num < epoch_length:
            try:
                if len(rpn_accuracy_rpn_monitor) == epoch_length and network.verbose:
                    mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor)) / len(
                        rpn_accuracy_rpn_monitor)
                    rpn_accuracy_rpn_monitor = []
                    print(
                        'Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(
                            mean_overlapping_bboxes, epoch_length))
                    if mean_overlapping_bboxes == 0:
                        print(
                            'RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

                X, Y, img_data = next(dataGenerator)
                # Xval, Yval, img_data_val = next(data_gen_val)
                width = X.shape[1]
                height = X.shape[0]

                try:
                    loss_rpn = self.model_rpn.train_on_batch(X, Y)
                except:
                    print("{} , {}".format(width,height))

                P_rpn = self.model_rpn.predict_on_batch(X)

                R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], network, use_regr=True, overlap_thresh=0.7,
                                           max_boxes=300)
                # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
                X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, network, self.class_mapping,labelName,width,height)


                if X2 is None:
                    rpn_accuracy_rpn_monitor.append(0)
                    rpn_accuracy_for_epoch.append(0)
                    continue
                neg_samples = np.where(Y1[0, :, -1] == 1)
                pos_samples = np.where(Y1[0, :, -1] == 0)

                if len(neg_samples) > 0:
                    neg_samples = neg_samples[0]
                else:
                    neg_samples = []

                if len(pos_samples) > 0:
                    pos_samples = pos_samples[0]
                else:
                    pos_samples = []

                rpn_accuracy_rpn_monitor.append(len(pos_samples))
                rpn_accuracy_for_epoch.append((len(pos_samples)))

                if network.num_rois > 1:
                    if len(pos_samples) < network.num_rois // 2:
                        selected_pos_samples = pos_samples.tolist()
                    else:
                        selected_pos_samples = np.random.choice(pos_samples, network.num_rois // 2,
                                                                replace=False).tolist()
                    if neg_samples.size > 0:
                        try:
                            selected_neg_samples = np.random.choice(neg_samples,
                                                                    network.num_rois - len(selected_pos_samples),
                                                                    replace=False).tolist()
                        except:
                            try:
                                selected_neg_samples = np.random.choice(neg_samples,
                                                                            network.num_rois - len(selected_pos_samples),
                                                                            replace=True).tolist()
                            except:
                                selected_neg_samples = np.random.choice(neg_samples,
                                                                            1,
                                                                            replace=True).tolist()

                        sel_samples = selected_pos_samples + selected_neg_samples
                    else:
                        sel_samples = selected_pos_samples
                else:
                    # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                    selected_pos_samples = pos_samples.tolist()
                    selected_neg_samples = neg_samples.tolist()
                    if np.random.randint(0, 2):
                        sel_samples = random.choice(neg_samples)
                    else:
                        sel_samples = random.choice(pos_samples)
                try:
                    loss_class = self.model_classifier.train_on_batch([X, X2[:, sel_samples, :]],
                                                                 [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])
                except:
                    print(sel_samples)

                losses[iter_num, 0] = loss_rpn[1]
                losses[iter_num, 1] = loss_rpn[2]

                losses[iter_num, 2] = loss_class[1]
                losses[iter_num, 3] = loss_class[2]
                losses[iter_num, 4] = loss_class[3]

                progbar.update(iter_num + 1,
                               [('rpn_cls', losses[iter_num, 0]), ('rpn_regr', losses[iter_num, 1]),
                                ('detector_cls', losses[iter_num, 2]), ('detector_regr', losses[iter_num, 3])])

                iter_num += 1

            except Exception as e:
                print("Exception: {}".format(e))
                continue

        loss_rpn_cls = np.mean(losses[:, 0])
        loss_rpn_regr = np.mean(losses[:, 1])
        loss_class_cls = np.mean(losses[:, 2])
        loss_class_regr = np.mean(losses[:, 3])
        class_acc = np.mean(losses[:, 4])
        self.history['rpn_cls_loss'][epoch_num] = loss_rpn_cls
        self.history['rpn_reg_loss'][epoch_num] = loss_rpn_regr
        self.history['cls_cls_loss'][epoch_num] = loss_class_cls
        self.history['cls_reg_loss'][epoch_num] = loss_class_regr
        self.history['loss'][epoch_num] = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
        self.history['accuracy'][epoch_num] = class_acc

        mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)

        print(
            'Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                mean_overlapping_bboxes))
        print('Train Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
        print('Train Loss RPN classifier: {}'.format(loss_rpn_cls))
        print('Train Loss RPN regression: {}'.format(loss_rpn_regr))
        print('Train Loss Detector classifier: {}'.format(loss_class_cls))
        print('Train Loss Detector regression: {}'.format(loss_class_regr))
        print('Elapsed time: {}'.format(time.time() - start_time))


    def testOneEpoch(self,epoch_num, epoch_length, dataGenerator, network, rpn_accuracy_rpn_monitor,labelName,testSet=False):
        progbar = utils.Progbar(epoch_length)
        print('Epoch {}/{}'.format(epoch_num, self.numEpochs))
        iternum = 0
        rpn_accuracy_for_epoch = []
        losses = np.zeros((epoch_length, 5))
        start_time = time.time()
        while iternum < epoch_length:
            try:
                X, Y, img_data = next(dataGenerator)
                width = X.shape[1]
                height = X.shape[0]
                loss_rpn = self.model_rpn.test_on_batch(X, Y)
                P_rpn = self.model_rpn.predict_on_batch(X)

                R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], network, use_regr=True,
                                              overlap_thresh=0.7, max_boxes=300)
                # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
                X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, network,
                                                                    self.class_mapping,labelName,width,height)

                if X2 is None:
                    rpn_accuracy_rpn_monitor.append(0)
                    rpn_accuracy_for_epoch.append(0)
                    continue

                neg_samples = np.where(Y1[0, :, -1] == 1)
                pos_samples = np.where(Y1[0, :, -1] == 0)

                if len(neg_samples) > 0:
                    neg_samples = neg_samples[0]
                else:
                    neg_samples = []

                if len(pos_samples) > 0:
                    pos_samples = pos_samples[0]
                else:
                    pos_samples = []

                rpn_accuracy_rpn_monitor.append(len(pos_samples))
                rpn_accuracy_for_epoch.append((len(pos_samples)))

                if network.num_rois > 1:
                    if len(pos_samples) < network.num_rois // 2:
                        selected_pos_samples = pos_samples.tolist()
                    else:
                        selected_pos_samples = np.random.choice(pos_samples, network.num_rois // 2,
                                                                replace=False).tolist()
                    if neg_samples.size > 0:
                        try:
                            selected_neg_samples = np.random.choice(neg_samples,
                                                                    network.num_rois - len(selected_pos_samples),
                                                                    replace=False).tolist()
                        except:
                            try:
                                selected_neg_samples = np.random.choice(neg_samples,
                                                                        network.num_rois - len(selected_pos_samples),
                                                                        replace=True).tolist()
                            except:
                                selected_neg_samples = np.random.choice(neg_samples,
                                                                        1,
                                                                        replace=True).tolist()

                        sel_samples = selected_pos_samples + selected_neg_samples
                    else:
                        sel_samples = selected_pos_samples
                else:
                    # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                    if np.random.randint(0, 2):
                        sel_samples = random.choice(neg_samples)
                    else:
                        sel_samples = random.choice(pos_samples)

                loss_class = self.model_classifier.test_on_batch([X, X2[:, sel_samples, :]],
                                                                [Y1[:, sel_samples, :],
                                                                 Y2[:, sel_samples, :]])

                losses[iternum, 0] = loss_rpn[1]
                losses[iternum, 1] = loss_rpn[2]

                losses[iternum, 2] = loss_class[1]
                losses[iternum, 3] = loss_class[2]
                losses[iternum, 4] = loss_class[3]

                progbar.update(iternum + 1,
                               [('rpn_cls', losses[iternum, 0]), ('rpn_regr', losses[iternum, 1]),
                                ('detector_cls', losses[iternum, 2]), ('detector_regr', losses[iternum, 3])])

                iternum += 1
            except Exception as e:
                print("Exception: {}".format(e))
                continue

        loss_rpn_cls = np.mean(losses[:, 0])
        loss_rpn_regr = np.mean(losses[:, 1])
        loss_class_cls = np.mean(losses[:, 2])
        loss_class_regr = np.mean(losses[:, 3])
        class_acc = np.mean(losses[:, 4])
        if not testSet:
            self.history['val_rpn_cls_loss'][epoch_num] = loss_rpn_cls
            self.history['val_rpn_reg_loss'][epoch_num] = loss_rpn_regr
            self.history['val_cls_cls_loss'][epoch_num] = loss_class_cls
            self.history['val_cls_reg_loss'][epoch_num] = loss_class_regr
            self.history['val_loss'][epoch_num] = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
            self.history['val_accuracy'][epoch_num] = class_acc

            mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(
                rpn_accuracy_for_epoch)
            print(
                'Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                    mean_overlapping_bboxes))
            print(
                'Validation Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
            print('Validation Loss RPN classifier: {}'.format(loss_rpn_cls))
            print('Validation Loss RPN regression: {}'.format(loss_rpn_regr))
            print('Validation Loss Detector classifier: {}'.format(loss_class_cls))
            print('Validation Loss Detector regression: {}'.format(loss_class_regr))
            print('Elapsed time: {}'.format(time.time() - start_time))
        else:
            self.results['rpn_cls_loss'][epoch_num] = loss_rpn_cls
            self.results['rpn_reg_loss'][epoch_num] = loss_rpn_regr
            self.results['cls_cls_loss'][epoch_num] = loss_class_cls
            self.results['cls_reg_loss'][epoch_num] = loss_class_regr
            self.results['loss'][epoch_num] = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
            self.results['accuracy'][epoch_num] = class_acc






    def train(self):
        if FLAGS.save_location == "":
            print("No save location specified. Please set flag --save_location")
        elif FLAGS.data_csv_file == "":
            print("No dataset specified. Please set flag --data_csv_file")
        else:
            self.saveLocation = FLAGS.save_location
            self.networkType = os.path.basename(os.path.dirname(self.saveLocation))
            self.dataCSVFile = pandas.read_csv(FLAGS.data_csv_file)
            self.numEpochs = FLAGS.num_epochs
            self.batch_size = FLAGS.batch_size
            self.learning_rate = FLAGS.learning_rate
            self.optimizer = 'sgd'
            self.loss_Function = 'mae'
            self.metrics = FLAGS.metrics.split(",")
            self.gClient = None
            self.selectedLabels = None
            self.balanceDataset = False
            self.patience = 3
            self.validation_percentage = FLAGS.validation_percentage
            network = Faster_RCNN.Faster_RCNN()
            foldDir = self.saveLocation
            if not os.path.exists(foldDir):
                os.mkdir(foldDir)

            labelName = "Tool bounding box" #This should be the label that will be used to train the network

            trainDataset, valDataset = self.loadData(self.validation_percentage,self.dataCSVFile)
            self.classes_count, self.class_mapping, trainDataset,imageMeans = self.getClassMapping(trainDataset,labelName)
            print("Class mapping: {}".format(self.class_mapping))
            print("Training class count: {}".format(self.classes_count))
            self.val_classes_count,_,valDataset, _ = self.getClassMapping(valDataset,labelName)
            print("Validation class count: {}".format(self.classes_count))

            if FLAGS.parser == 'pascal_voc':
                from keras_frcnn.pascal_voc_parser import get_data
            elif FLAGS.parser == 'simple':
                from keras_frcnn.simple_parser import get_data
            else:
                raise ValueError("Command line option parser must be one of 'pascal_voc' or 'simple'")

            # pass the settings from the command line, and persist them in the config object
            network.img_channel_mean = imageMeans

            network.im_size=224

            #C.model_path = options.output_weight_path #foldDir
            network.num_rois = int(100)

            # check if weight path was passed via command line

            network.base_net_weights = nn.get_weight_path()

            if 'bg' not in self.classes_count:
                self.classes_count['bg'] = 0
                self.class_mapping['bg'] = len(self.class_mapping)

            network.class_mapping = self.class_mapping

            config_output_filename = os.path.join(foldDir, 'config.pickle')

            with open(config_output_filename, 'wb') as config_f:
                pickle.dump(network, config_f)
                print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(
                    config_output_filename))

            print('Num train samples {}'.format(len(trainDataset.index)))
            print('Num val samples {}'.format(len(valDataset.index)))

            data_gen_train = data_generators.get_anchor_gt(trainDataset, self.classes_count, network, nn.get_img_output_length, labelName,
                                                           mode='train')
            data_gen_val = data_generators.get_anchor_gt(valDataset, self.val_classes_count, network, nn.get_img_output_length, labelName,
                                                         mode='val')

            self.model_rpn,self.model_classifier,self.model_all = network.createModel(len(self.class_mapping))
            num_anchors = len(network.anchor_box_scales) * len(network.anchor_box_ratios)

            try:
                print('loading weights from {}'.format(network.base_net_weights))
                self.model_rpn.load_weights(network.base_net_weights, by_name=True)
                self.model_classifier.load_weights(network.base_net_weights, by_name=True)
            except:
                print('Could not load pretrained model weights. Weights can be found in the keras application folder \
            		https://github.com/fchollet/keras/tree/master/keras/applications')

            optimizer = Adam(lr=1e-5)
            optimizer_classifier = Adam(lr=1e-5)
            self.model_rpn.compile(optimizer=optimizer,
                                   loss=[lossFunctions.rpn_loss_cls(num_anchors), lossFunctions.rpn_loss_regr(num_anchors)])
            self.model_classifier.compile(optimizer=optimizer_classifier,
                                     loss=[lossFunctions.class_loss_cls, lossFunctions.class_loss_regr(len(self.classes_count) - 1)],
                                     metrics={'dense_class_{}'.format(len(self.classes_count)): 'accuracy'})
            self.model_all.compile(optimizer='sgd', loss='mae')

            epoch_length = len(trainDataset.index)
            # epoch_length  = 200
            val_epoch_length = len(valDataset.index)
            # val_epoch_length = 100

            losses = np.zeros((epoch_length, 5))
            rpn_accuracy_rpn_monitor = []
            rpn_accuracy_rpn_monitor_val = []
            rpn_accuracy_rpn_monitor_test = []


            best_loss = np.Inf
            best_acc = -1

            val_loss_decreasing = True
            val_acc_increasing = True

            epoch_num=0
            numEpochsWithoutImprovement = 0
            if os.path.exists(os.path.join(foldDir,"logFile.txt")):
                with open(os.path.join(foldDir,"logFile.txt"),'r') as logFile:
                    completedEpochs = logFile.readline()
                    self.model_rpn.load_weights(os.path.join(foldDir, 'tempWeights.hdf5'), by_name=True)
                    self.model_classifier.load_weights(os.path.join(foldDir, 'tempWeights.hdf5'), by_name=True)
                completedEpochs = completedEpochs.replace("Epochs completed: ","")
                epoch_num = int(completedEpochs)
                epoch_num += 1
            if os.path.exists(os.path.join(foldDir,"history.csv")):
                self.history = pandas.read_csv(os.path.join(foldDir, "history.csv"))
            else:
                self.history = pandas.DataFrame(columns=['rpn_cls_loss',
                                                         'rpn_reg_loss',
                                                         'cls_cls_loss',
                                                         'cls_reg_loss',
                                                         'loss',
                                                         'accuracy',
                                                         'val_rpn_cls_loss',
                                                         'val_rpn_reg_loss',
                                                         'val_cls_cls_loss',
                                                         'val_cls_reg_loss',
                                                         'val_loss',
                                                         'val_accuracy'])
                for col in self.history.columns:
                    self.history[col] = [None for i in range(self.numEpochs)]

            while (val_loss_decreasing and val_acc_increasing and epoch_num < self.numEpochs):
            #for epoch_num in range(self.numEpochs):
                self.trainOneEpoch(epoch_num,epoch_length,data_gen_train,network,rpn_accuracy_rpn_monitor,labelName)
                self.testOneEpoch(epoch_num,val_epoch_length,data_gen_val,network,rpn_accuracy_rpn_monitor_val,labelName)
                curr_loss = self.history["val_loss"][epoch_num]
                curr_accuracy = self.history["val_accuracy"][epoch_num]
                epoch_num+=1
                if curr_loss < best_loss:
                    print(
                        'Total loss decreased from {} to {}, saving weights'.format(best_loss, curr_loss))
                    best_loss = curr_loss
                    self.model_all.save_weights(os.path.join(foldDir, 'frcnn.hdf5'))
                    if curr_accuracy > best_acc:
                        best_acc = curr_accuracy
                    numEpochsWithoutImprovement = 0
                elif curr_accuracy > best_acc:
                    print('Total accuracy increased from {} to {}, saving weights'.format(best_acc,
                                                                                          curr_accuracy))
                    best_acc = curr_accuracy
                    self.model_all.save_weights(os.path.join(foldDir, 'frcnn.hdf5'))
                    numEpochsWithoutImprovement = 0
                else:
                    numEpochsWithoutImprovement +=1
                    if numEpochsWithoutImprovement >= self.patience:
                        print("{} epochs without improvement. EARLY STOPPING".format(self.patience))
                        val_acc_increasing = False
                        val_loss_decreasing = False
                self.history.to_csv(os.path.join(foldDir,"history.csv"),index=False)
                self.model_all.save_weights(os.path.join(foldDir, 'tempWeights.hdf5'))
                with open(os.path.join(foldDir,"logFile.txt"),"w") as logFile:
                    logFile.write("Epochs completed: {}".format(epoch_num - 1))
                gc.collect()

            self.model_rpn.load_weights(os.path.join(foldDir, 'frcnn.hdf5'), by_name=True)
            self.model_classifier.load_weights(os.path.join(foldDir, 'frcnn.hdf5'), by_name=True)
            self.saveTrainingInfo(foldDir,self.history)
            self.saveTrainingPlot(foldDir,self.history,"loss")
            for metric in self.metrics:
                self.saveTrainingPlot(foldDir,self.history,metric)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--save_location',
      type=str,
      default='',
      help='Name of the directory where the models and results will be saved'
  )
  parser.add_argument(
      '--data_csv_file',
      type=str,
      default='',
      help='Path to the csv file containing locations for all data used in training'
  )
  parser.add_argument(
      '--validation_percentage',
      type=float,
      default=0.3,
      help='Percent of data to be used for validation'
  )
  parser.add_argument(
      '--num_epochs',
      type=int,
      default=25,
      help='number of epochs used in training'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=16,
      help='type of output your model generates'
  )
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.0001,
      help='Learning rate used in training'
  )
  parser.add_argument(
      '--loss_function',
      type=str,
      default='mae',
      help='Name of the loss function to be used in training (see keras documentation).'
  )
  parser.add_argument(
      '--metrics',
      type=str,
      default='accuracy,rpn_cls_loss,rpn_reg_loss,cls_cls_loss,cls_reg_loss',
      help='Metrics used to evaluate model.'
  )
  parser.add_argument(
      '--parser',
      type=str,
      default='simple',
      help='Metrics used to evaluate model.'
  )
FLAGS, unparsed = parser.parse_known_args()
tm = Train_Faster_RCNN()
tm.train()
