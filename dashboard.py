from visdom import Visdom
import numpy as np
import utils
import pandas as pd
import argparse
import pickle
import os

parser = argparse.ArgumentParser(description='Consistency driven Multi-Lable Image Classification')
parser.add_argument('--name', default='', type=str, help='Performance filename')
args = parser.parse_args()

plotter = utils.VisdomPlotter(env_name=args.name)

paramFile = os.path.join('runs', args.name, "parameters.csv")
perfFile = os.path.join('runs', args.name, "performance.csv")
mediaFile = os.path.join('runs', args.name, "media.pkl")

# Display parameters
paramsDict = {}
for line in open(paramFile):
    key, value = line.split(',')
    paramsDict.update({key: value})

plotter.argsTile(paramsDict)

# Display performance
performanceDF = pd.read_csv(perfFile, sep=',', header=0)
trainDF = performanceDF[(performanceDF['trainFlag'] == 'train')]
testDF = performanceDF[(performanceDF['trainFlag'] == 'test')]
# Plotting Train
plotter.plotPerformance('loss', 'train', trainDF.epoch, trainDF.loss)
plotter.plotPerformance('Per-class Precision', 'train', trainDF.epoch, trainDF.pcPrecision)
plotter.plotPerformance('Per-class Recall', 'train', trainDF.epoch, trainDF.pcRecall)
plotter.plotPerformance('Overall Precision', 'train', trainDF.epoch, trainDF.ovPrecision)
plotter.plotPerformance('Overall Recall', 'train', trainDF.epoch, trainDF.ovRecall)
plotter.plotPerformance('0/1 Score', 'train', trainDF.epoch, trainDF.zeroOneScore)
plotter.plotPerformance('Per-class F1', 'train', trainDF.epoch, trainDF.pcF1)
plotter.plotPerformance('Overall F1', 'train', trainDF.epoch, trainDF.ovF1)
# Plotting Test
plotter.plotPerformance('loss', 'test', testDF.epoch, testDF.loss)
plotter.plotPerformance('Per-class Precision', 'test', testDF.epoch, testDF.pcPrecision)
plotter.plotPerformance('Per-class Recall', 'test', testDF.epoch, testDF.pcRecall)
plotter.plotPerformance('Overall Precision', 'test', testDF.epoch, testDF.ovPrecision)
plotter.plotPerformance('Overall Recall', 'test', testDF.epoch, testDF.ovRecall)
plotter.plotPerformance('0/1 Score', 'test', testDF.epoch, testDF.zeroOneScore)
plotter.plotPerformance('Per-class F1', 'test', testDF.epoch, testDF.pcF1)
plotter.plotPerformance('Overall F1', 'test', testDF.epoch, testDF.ovF1)
# Plotting the media
if os.path.exists(mediaFile):
    mediaArray = pickle.load(open(mediaFile, "rb" ))
    plotter.showImage(mediaArray)