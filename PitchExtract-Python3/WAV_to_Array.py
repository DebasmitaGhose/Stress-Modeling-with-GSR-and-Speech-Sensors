# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 19:18:24 2018

@author: User
"""

import myWave
import dspUtil
import numpy
import copy
import generalUtility
fName = 'C:/Users/User/Documents/UMass Amherst/Semester 2/COMPSCI 590U - Mobile and Ubiquitous Computing/Assignments/Assignment 3/Speech_Data/subject_1_stress.wav'
# http://en.wikipedia.org/wiki/Wilhelm_scream
# http://www.youtube.com/watch?v=Zf8aBFTVNEU
# load the input file
# data is a list of numpy arrays, one for each channel
numChannels, numFrames, fs, data = myWave.readWaveFile(fName)
print(numChannels)
print(numFrames)
print(fs)
print(len(data[1]))
print("1")
# normalize the left channel, leave the right channel untouched
data[0] = dspUtil.normalize(data[0])
print("2")
# just for kicks, reverse (i.e., time-invert) all channels
for chIdx in range(numChannels):
    print("3")
    n = len(data[chIdx])
    dataTmp = copy.deepcopy(data[chIdx])
    for i in range(n):
        data[chIdx][i] = dataTmp[n - (i + 1)]
# save the normalized file (both channels)
# this is the explicit code version, to make clear what we're doing. since we've
# treated the data in place, we could simple write: 
# myWave.writeWaveFile(data, outputFileName, fs) and not declare dataOut
print("4")
dataOut = [data[0], data[1]] 
print(data[0].shape)
x = numpy.array([])
x = data[1]
fileNameOnly = generalUtility.getFileNameOnly(fName)
outputFileName = fileNameOnly + "_processed.wav"
myWave.writeWaveFile(dataOut, outputFileName, fs)

pitch = {}
pitch = dspUtil.calculateF0(x[0:1000],fs)
print(pitch)
   