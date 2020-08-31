from visdom import Visdom
import numpy as np
import os, shutil
import torch
import warnings
warnings.simplefilter(action='ignore', category=ImportWarning)
from sklearn import metrics
from collections import defaultdict
import itertools
from sklearn.metrics import average_precision_score
from sklearn.metrics import auc
from sklearn.metrics import f1_score
import subprocess
import csv
import pickle

class VisdomPlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
        self.paramList = {}
    def argsTile(self, argsDict):
        self.paramList = self.viz.text('<b>Training Parameters:</b>\n', env=self.env, opts=dict(width=220,height=320))
        for key, value in argsDict.items():
            self.viz.text(str(key) + ' = ' + str(value) + '\n', env=self.env, win=self.paramList, append=True)
    def plot(self, var_name, split_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env,
                opts=dict(legend=[split_name], title=var_name, xlabel='Epochs', ylabel=var_name))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update='append')
    def showImage(self, imageTensor):
        # self.viz.image(imageTensor, win=self.images, env=self.env, opts=dict(title='Original and Reconstructed', caption='How random.'),)
        self.viz.images(imageTensor, win=self.images, env=self.env, opts=dict(title='Original and Reconstructed', caption='How random.', nrow=2),)
    def plotPerformance(self, var_name, split_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=x, Y=y, env=self.env,
                opts=dict(legend=[split_name], title=var_name, xlabel='Epochs', ylabel=var_name))
        else:
            self.viz.line(X=x, Y=y, env=self.env, win=self.plots[var_name], name=split_name, update='append')

class diskWriter(object):
    """Writes to CSV"""
    def __init__(self, directory):
        root = 'runs'
        if os.path.exists(os.path.join(root, directory)):
            shutil.rmtree(os.path.join(root, directory))
        self.performanceFilename = os.path.join(root, directory, "performance.csv")
        self.parametersFilename = os.path.join(root, directory, "parameters.csv")
        self.mediaFile = os.path.join(root, directory, "media.pkl")
        os.makedirs(os.path.join(root, directory))
        open(self.performanceFilename, 'a').close()
        open(self.parametersFilename, 'a').close()
    def writePerformance(self, datalist):
        with open(self.performanceFilename, "a") as file:
            file.write(','.join(map(str, datalist)))
            file.write('\n')
    def writeParameters(self, paramsDict):
        with open(self.parametersFilename, "a") as file:
            for key, value in paramsDict.items():
                file.write(str(key) + ', ' + str(value) + '\n')
    def writeMedia(self, mediaArray):
        pickle.dump(mediaArray, open(self.mediaFile, "wb" ))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def load_checkpoint(resume):
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        startEpoch = checkpoint['epoch']
        perfMetric = checkpoint['perfMetric']
        stateDict = checkpoint['stateDict']
        print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
        return startEpoch, perfMetric, stateDict
    else:
        print("=> no checkpoint found at '{}'".format(resume))


def save_checkpoint(state, is_best, name):
    """Saves checkpoint to disk"""
    saveDir = '/home/SharedData/saurabh/MLC/models/%s/'%(name)
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    filename = saveDir + 'checkpoint.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, saveDir + 'model_best.pth.tar')

def batchPerformance(predictedLabels, labels, totalCorrectPC, totalTruePC, totalPredPC, exactMatches):
    trueInBatch = labels.sum(dim=0)
    predInBatch = predictedLabels.sum(dim=0)
    correctInBatch = (labels * predictedLabels).sum(dim=0)
    totalTruePC = torch.add(totalTruePC, trueInBatch)
    totalPredPC = torch.add(totalPredPC, predInBatch)
    totalCorrectPC = torch.add(totalCorrectPC, correctInBatch)
    for i in range(labels.size(0)):
        exactMatches += torch.equal(predictedLabels[i], labels[i])
    return totalCorrectPC, totalTruePC, totalPredPC, exactMatches

def measurePerformance(totalCorrectPC, totalPredPC, totalTruePC, classCount, exactMatches, numSamples):
    pcPrecisions = torch.div(totalCorrectPC, totalPredPC)
    pcPrecision = pcPrecisions.sum()/classCount
    pcRecalls = torch.div(totalCorrectPC, totalTruePC)
    pcRecall = pcRecalls.sum()/classCount
    pcF1s = 2*torch.div(torch.mul(pcPrecisions, pcRecalls), torch.add(pcPrecisions, pcRecalls))
    pcF1 = pcF1s.sum()/classCount
    totalCorrectOV = totalCorrectPC.sum().item()
    ovPrecision = totalCorrectOV / totalPredPC.sum().item()     # totalCorrectOV/totalPredOV
    ovRecall = totalCorrectOV / totalTruePC.sum().item()        # totalCorrectOV/totalTrueOV
    ovF1 = 2*(ovPrecision * ovRecall)/(ovPrecision + ovRecall)
    zeroOneScore = exactMatches/numSamples         # exactMatches/totalSamples
    return pcPrecision, pcRecall, pcF1, ovPrecision, ovRecall, ovF1, zeroOneScore

def verbose(predictedLabels, labels):
    outLoc = set((np.argpartition(predictedLabels[0].detach().cpu().numpy(), -5)[-5:]).flatten())
    labLoc = set((np.array(np.nonzero(labels[0].detach().cpu().numpy()))).flatten())
    caught = outLoc.intersection(labLoc)
    missed = labLoc.symmetric_difference(caught)
    print('OutLoc = ' + str(outLoc))
    print('LabLoc = ' + str(labLoc))
    print('Caught = ' + str(caught))
    print('Missed = ' + str(missed))