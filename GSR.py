# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 01:54:13 2018

@author:Debasmita Ghose
"""

import numpy as np
import pandas as pd
import glob
from numpy.lib.stride_tricks import as_strided
import scipy as sc
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy import signal

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

import warnings


####################################################### Data Read ###########################################################

#Read data from all files in a data-frame and convert to a numpy array

#----------------------------- GSR ----------------------------------------------------------
path =r'C:/Users/User/Documents/UMass Amherst/Semester 2/COMPSCI 590U - Mobile and Ubiquitous Computing/Assignments/Assignment 3/Raw Data' # use your path
allFiles = glob.glob(path + "/*.csv")
frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    list_.append(df)
frame = pd.concat(list_)
data = frame.values # convert data frame to numpy array
EDA = data[:,6] # extract the EDA value

#----------------------------- Speech ---------------------------------------------------------
path =r'C:/Users/User/Documents/UMass Amherst/Semester 2/COMPSCI 590U - Mobile and Ubiquitous Computing/Assignments/Assignment 3/Speech_Data/Speech-TimeSeries' # use your path
allFiles = glob.glob(path + "/*.csv")
frame = pd.DataFrame()
list_ = []
list_1 = np.array([])
df_1 = np.array([])
for file_ in allFiles:
    df = pd.read_csv(file_,sep = ',',index_col=None, header=1)
    df_1 = df.values
    df1 = np.array(df_1)
    list_.append(df_1)
list_1 = np.vstack(list_)
pitch = list_1[:,0] #extract pitch value
###################################################### Labels Read ##########################################################

#Read labels from all files in a data-frame and convert to a numpy array

#----------------------------- GSR -------------------------------------------------------------
path =r'C:/Users/User/Documents/UMass Amherst/Semester 2/COMPSCI 590U - Mobile and Ubiquitous Computing/Assignments/Assignment 3/Raw Data' # use your path
allFiles = glob.glob(path + "/*.csv")
frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    list_.append(df)
frame = pd.concat(list_)
data = frame.values # convert data frame to numpy array
EDA_labels = data[:,7] # extract the EDA labels

#----------------------------- Speech -------------------------------------------------------------
path =r'C:/Users/User/Documents/UMass Amherst/Semester 2/COMPSCI 590U - Mobile and Ubiquitous Computing/Assignments/Assignment 3/Speech_Data/Speech-Labels' # use your path
allFiles = glob.glob(path + "/*.csv")
frame = pd.DataFrame()
list_ = []
list_1 = np.array([])
df_1 = np.array([])
for file_ in allFiles:
    df = pd.read_csv(file_,sep = ',',index_col=None, header=1)
    df_1 = df.values
    df1 = np.array(df_1)
    list_.append(df_1)
list_1 = np.vstack(list_)
pitch_labels = list_1[:,0] #extract pitch labels
        
##################################################### Functions ##############################################################

#function to convert time to decimals
def to_seconds(s):
    hr, min = [float(x) for x in s.split(':')]
    return hr*3600 + min*60 

#Run the script only if data is changed
'''
time_conv = np.array(len(EDA))
time = data[:,0]
for i in range(0,len(EDA)):
    time_val = to_seconds(time[i])
    print(i)
    time_conv = np.append(time_conv, time_val)
np.savetxt("time_converted.csv",time_conv,delimiter=',')
#print(time_conv)
'''

#function to window the data with overlap
def windowed_view(arr, window, overlap):
    arr = np.asarray(arr)
    window_step = window - overlap
    new_shape = arr.shape[:-1] + ((arr.shape[-1] - overlap) // window_step,
                                  window)
    new_strides = (arr.strides[:-1] + (window_step * arr.strides[-1],) +
                   arr.strides[-1:])
    return as_strided(arr, shape=new_shape, strides=new_strides)

#load pre-computed time - update if GSR data is changed
time = np.loadtxt("time_converted.csv",delimiter=',')   
    
###################################### GSR Feature Extraction ################################################################

EDA_peaks_number = np.array([])
EDA_peaks_mean = np.array([])

#window size of 10 with an overlap of 5
EDA_windowed = windowed_view(EDA,10,5)
EDA_windowed_flattened = np.array(EDA_windowed.flatten())
time_windowed = windowed_view(time,10,5)

#feature extraction - GSR
EDA_windowed_avg = np.mean(EDA_windowed, axis=-1)
EDA_windowed_max = np.max(EDA_windowed, axis=-1)
EDA_windowed_min = np.min(EDA_windowed, axis=-1)
EDA_slope = ((EDA_windowed*time_windowed).mean(axis=1) - EDA_windowed.mean()*time_windowed.mean(axis=1)) / ((EDA_windowed**2).mean() - (time_windowed.mean())**2)
for i in range(0,len(EDA_windowed)):
    EDA_peak_indices = (argrelextrema(EDA_windowed[i],np.greater)[0])
    EDA_peaks_number = np.append(EDA_peaks_number,(len(EDA_peak_indices)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        EDA_peaks_mean = np.append(EDA_peaks_mean,np.nan_to_num(np.mean(EDA_windowed[EDA_peak_indices])))
EDA_peaks_25 = np.percentile(EDA_windowed,25,axis=-1)
EDA_peaks_50 = np.percentile(EDA_windowed,50,axis=-1)
EDA_peaks_75 = np.percentile(EDA_windowed,75,axis=-1)

#arranging features into feature vector
EDA_windowed_avg = np.vstack(EDA_windowed_avg)
EDA_windowed_max = np.vstack(EDA_windowed_max)
EDA_windowed_min = np.vstack(EDA_windowed_min)
EDA_slope = np.vstack(EDA_slope)
EDA_peaks_mean = np.vstack(EDA_peaks_mean)
EDA_peaks_number = np.vstack(EDA_peaks_number)
EDA_peaks_25 = np.vstack(EDA_peaks_25)
EDA_peaks_50 = np.vstack(EDA_peaks_50)
EDA_peaks_75 = np.vstack(EDA_peaks_75)

x_GSR = np.concatenate((EDA_windowed_avg,EDA_windowed_max,EDA_windowed_min,EDA_slope,EDA_peaks_mean,EDA_peaks_number,EDA_peaks_25,EDA_peaks_50,EDA_peaks_75),axis=1) # 60441 x 4
print("works GSR!")

#window the labels and select the max value from each window
EDA_labels_windowed = windowed_view(EDA_labels,10,5)
EDA_labels_max = np.max(EDA_labels_windowed,axis=-1)

y_GSR = np.vstack(EDA_labels_max) # 60441 x 1

################################### Speech Feature Extraction ################################################################

#window size of 10 with overlap of 5 - data available every 10ms
pitch_windowed = windowed_view(pitch,10,5)
pitch_windowed_flattened = np.array(pitch_windowed.flatten())
time_gen = np.arange(0,len(pitch_windowed))
time_gen = np.vstack(time_gen)

#feature extraction - pitch
pitch_windowed_std = np.std(pitch_windowed,axis=1)
pitch_slope = ((pitch_windowed*time_gen).mean(axis=1) - pitch_windowed.mean()*time_gen.mean(axis=1)) / ((pitch_windowed**2).mean() - (time_gen.mean())**2)
pitch_windowed_skewness = sc.stats.skew(pitch_windowed,axis=1)
pitch_windowed_kurtosis = sc.stats.kurtosis(pitch_windowed,axis=1)
pitch_windowed_avg = np.mean(pitch_windowed, axis=-1)
pitch_windowed_max = np.max(pitch_windowed, axis=-1)
pitch_windowed_min = np.min(pitch_windowed, axis=-1)

#arranging features into feature vector
pitch_windowed_std = np.vstack(pitch_windowed_std)
pitch_slope = np.vstack(pitch_slope)
pitch_windowed_skewness = np.vstack(pitch_windowed_skewness)
pitch_windowed_kurtosis = np.vstack(pitch_windowed_kurtosis)
pitch_windowed_avg = np.vstack(pitch_windowed_avg)
pitch_windowed_max = np.vstack(pitch_windowed_max)
pitch_windowed_min = np.vstack(pitch_windowed_min)

x_speech = np.concatenate((pitch_windowed_std,pitch_slope,pitch_windowed_skewness,pitch_windowed_kurtosis,pitch_windowed_avg,pitch_windowed_max,pitch_windowed_min),axis=-1) #202951 x 7

#window the labels and select the max value from each window
pitch_labels_windowed = windowed_view(pitch_labels,10,5)
pitch_labels_max = np.max(pitch_labels_windowed,axis=-1)

y_speech = np.vstack(pitch_labels_max) # 202951 x 1
print("works speech!")

##################################################### Feature Selection - GSR ###########################################################
print("###################################################################################################")
print("Feature Selection - GSR")
print("###################################################################################################")
      
#split the data into train and test sets - 70% for training
x_GSR_train,x_GSR_test,y_GSR_train,y_GSR_test = train_test_split(x_GSR,y_GSR,test_size=0.3,random_state = 1);
y_GSR_train = np.ravel(y_GSR_train)
y_GSR_test = np.ravel(y_GSR_test)

features = ["Average","Minimum","Maximum","Slope","Mean Peak Height","No.of Peaks","25th Perc.","50th Perc.","75th Perc."]
num_features = 9

#single feature
score_single = np.array([])
best_score = 0
for i in range(0,num_features):
    #select feature
    x_train = x_GSR_train[:,i]
    x_train = x_train.reshape(-1,1)
    x_test = x_GSR_test[:,i]
    x_test = x_test.reshape(-1,1)
    #build classifier and predict
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_train,y_GSR_train)
    y_pred = clf.predict(x_test)
    Score_f1 = f1_score(y_GSR_test,y_pred)
    print(features[i]," : ",Score_f1)
    print("------------------------")
    score_single = np.append(score_single,Score_f1)
    if Score_f1>best_score:
        best_score = Score_f1
        GSR_best_index = i
        best_feature = features[i]
print("*******************************************")
print("The best feature is: ", best_feature)
print("F1 Score = ", best_score)
print("*******************************************")

indices = np.arange(num_features)
plt.figure(1)
plt.bar(indices,score_single)
plt.xticks(indices,features)
plt.ylabel('F1 Score')
plt.xlabel('Features')
plt.title('F1 Score for a Decision Tree Classifier for GSR Features Trained for ONE feature at a time')

best_combination = 0
count = 0
score_double = np.array([])
#two features
for i in range(0,num_features-1):
    for j in range(i+1,num_features):
        #select features
        x_train_1 =x_GSR_train[:,i]
        x_train_1 = np.vstack(x_train_1)
        x_train_2 =x_GSR_train[:,j]
        x_train_2 = np.vstack(x_train_2)
        x_train = np.concatenate((x_train_1,x_train_2), axis = -1)
        x_test_1 =x_GSR_test[:,i]
        x_test_1 = np.vstack(x_test_1)
        x_test_2 =x_GSR_test[:,j]
        x_test_2 = np.vstack(x_test_2)
        x_test = np.concatenate((x_test_1,x_test_2), axis = -1)
        #build classifier and predict
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(x_train,y_GSR_train)
        y_pred = clf.predict(x_test)
        Score_f1 = f1_score(y_GSR_test,y_pred)
        print(features[i]," & ", features[j], " : ", Score_f1)
        print("------------------------------------------------")
        score_double = np.append(score_double,Score_f1)
        count = count+1
        if Score_f1>best_combination:
            best_combination = Score_f1
            best_feature_1 = features[i]
            GSR_best_index_1 = i
            best_feature_2 = features[j]
            GSR_best_index_2 = j
        
print("*******************************************************************************")
print("The best combination of features is: ", best_feature_1, " & ", best_feature_2)
print("F1 Score = ", best_combination)
print("*******************************************************************************")

indices = np.arange(count)
plt.figure(2)
plt.bar(indices,score_double)
plt.ylabel('F1 Score')
plt.xlabel('Features')
plt.title('F1 Score for a Decision Tree Classifier for GSR Features Trained for TWO features at a time')
##################################################### Feature Selection - Speech #########################################################
print("###################################################################################################")
print("Feature Selection - Speech")
print("###################################################################################################")
      
#split the data into train and test sets - 70% for training
x_speech_train,x_speech_test,y_speech_train,y_speech_test = train_test_split(x_speech,y_speech,test_size=0.3,random_state = 1);
y_speech_train = np.ravel(y_speech_train)
y_speech_test = np.ravel(y_speech_test)

features = ["Std.Deviation","Slope","Skewness","Kurtosis","Average","Maximum","Minimum"]
num_features = 7

#single feature
score_single = np.array([])
best_score = 0
for i in range(0,num_features):
    #select feature
    x_train = x_speech_train[:,i]
    x_train = x_train.reshape(-1,1)
    x_test = x_speech_test[:,i]
    x_test = x_test.reshape(-1,1)
    #build classifier and predict
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_train,y_speech_train)
    y_pred = clf.predict(x_test)
    Score_f1 = f1_score(y_speech_test,y_pred)
    print(features[i]," : ",Score_f1)
    print("-------------------------")
    score_single = np.append(score_single,Score_f1)
    if Score_f1>best_score:
        best_score = Score_f1
        speech_best_index = i
        best_feature = features[i]
print("**************************************")
print("The best feature is: ", best_feature)
print("F1 Score = ", best_score)
print("**************************************")

indices = np.arange(num_features)
plt.figure(3)
plt.bar(indices,score_single)
plt.xticks(indices,features)
plt.ylabel('F1 Score')
plt.xlabel('Features')
plt.title('F1 Score for a Decision Tree Classifier for Speech Features Trained for ONE feature at a time')

best_combination = 0
count = 0
score_double = np.array([])
#two features
for i in range(0,num_features-1):
    for j in range(i+1,num_features):
        #select features
        x_train_1 =x_speech_train[:,i]
        x_train_1 = np.vstack(x_train_1)
        x_train_2 =x_speech_train[:,j]
        x_train_2 = np.vstack(x_train_2)
        x_train = np.concatenate((x_train_1,x_train_2), axis = -1)
        x_test_1 =x_speech_test[:,i]
        x_test_1 = np.vstack(x_test_1)
        x_test_2 =x_speech_test[:,j]
        x_test_2 = np.vstack(x_test_2)
        x_test = np.concatenate((x_test_1,x_test_2), axis = -1)
        #build classifier and predict
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(x_train,y_speech_train)
        y_pred = clf.predict(x_test)
        Score_f1 = f1_score(y_speech_test,y_pred)
        print(features[i]," & ", features[j], " : ", Score_f1)
        print("-----------------------------------------------")
        score_double = np.append(score_double,Score_f1)
        count = count+1
        if Score_f1>best_combination:
            best_combination = Score_f1
            best_feature_1 = features[i]
            speech_best_index_1 = i
            best_feature_2 = features[j]
            speech_best_index_2 = j
print("*****************************************************************************")
print("The best combination of features is: ", best_feature_1, " & ", best_feature_2)
print("F1 Score = ", best_combination)
print("*****************************************************************************")
indices = np.arange(count)
plt.figure(4)
plt.bar(indices,score_double)
plt.ylabel('F1 Score')
plt.xlabel('Features')
plt.title('F1 Score for a Decision Tree Classifier for Speech Features Trained for TWO features at a time')

############################################################# Sensor Fusion ##############################################################

print("######################################################################################")
print("Best two GSR Features")
#Only GSR Features - best 2 features
x_train_1 = x_GSR_train[:,GSR_best_index_1]
x_train_1 = np.vstack(x_train_1)
x_train_2 = x_GSR_train[:,GSR_best_index_2]
x_train_2 = np.vstack(x_train_2)
x_train = np.concatenate((x_train_1,x_train_2),axis = -1)
x_test_1 = x_GSR_test[:,GSR_best_index_1]
x_test_1 = np.vstack(x_test_1)
x_test_2 = x_GSR_test[:,GSR_best_index_2]
x_test_2 = np.vstack(x_test_2)
x_test = np.concatenate((x_test_1,x_test_2),axis = -1)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train,y_GSR_train)
y_pred = clf.predict(x_test)
Score_f1 = f1_score(y_GSR_test,y_pred)
Score_recall = recall_score(y_GSR_test,y_pred)
Score_precision = precision_score(y_GSR_test,y_pred)
Score_accuracy = accuracy_score(y_GSR_test,y_pred)
print("F1 Score = ", Score_f1)
print("Recall = ", Score_recall)
print("Precision = ", Score_precision)
print("Accuracy = ", Score_accuracy)

print("######################################################################################")
print("Best two Speech Features")
#Only speech Features - best 2 features
x_train_1 = x_speech_train[:,speech_best_index_1]
x_train_1 = np.vstack(x_train_1)
x_train_2 = x_speech_train[:,speech_best_index_2]
x_train_2 = np.vstack(x_train_2)
x_train = np.concatenate((x_train_1,x_train_2),axis = -1)
x_test_1 = x_speech_test[:,speech_best_index_1]
x_test_1 = np.vstack(x_test_1)
x_test_2 = x_speech_test[:,speech_best_index_2]
x_test_2 = np.vstack(x_test_2)
x_test = np.concatenate((x_test_1,x_test_2),axis = -1)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train,y_speech_train)
y_pred = clf.predict(x_test)
Score_f1 = f1_score(y_speech_test,y_pred)
Score_recall = recall_score(y_speech_test,y_pred)
Score_precision = precision_score(y_speech_test,y_pred)
Score_accuracy = accuracy_score(y_speech_test,y_pred)
print("F1 Score = ", Score_f1)
print("Recall = ", Score_recall)
print("Precision = ", Score_precision)
print("Accuracy = ", Score_accuracy)

print("######################################################################################")
print("Speech and GSR Features")
print('-----------------------------------------------------')
print("Best of speech and GSR")
print('-----------------------------------------------------')
#speech and GSR Features - best of each
x_train_1 = x_GSR_train[:,GSR_best_index]
x_train_1 = np.vstack(x_train_1)
x_train_2 = x_speech_train[0:len(x_train_1),speech_best_index]
x_train_2 = np.vstack(x_train_2)
x_train = np.concatenate((x_train_1,x_train_2), axis = -1)
x_test_1 = x_GSR_test[:,GSR_best_index]
x_test_1 = np.vstack(x_test_1)
x_test_2 = x_speech_test[0:len(x_test_1),speech_best_index]
x_test_2 = np.vstack(x_test_2)
x_test = np.concatenate((x_test_1,x_test_2),axis = -1)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train,y_GSR_train)
y_pred = clf.predict(x_test)
Score_f1 = f1_score(y_GSR_test,y_pred)
Score_recall = recall_score(y_GSR_test,y_pred)
Score_precision = precision_score(y_GSR_test,y_pred)
Score_accuracy = accuracy_score(y_GSR_test,y_pred)
print("F1 Score = ", Score_f1)
print("Recall = ", Score_recall)
print("Precision = ", Score_precision)
print("Accuracy = ", Score_accuracy)

print("######################################################################################")
print('-----------------------------------------------------')
print("Best two of speech and GSR")
print('-----------------------------------------------------')
#speech and GSR Features - best two of each
x_train_1 = x_GSR_train[:,GSR_best_index_1]
x_train_1 = np.vstack(x_train_1)
x_train_2 = x_GSR_train[:,GSR_best_index_2]
x_train_2 = np.vstack(x_train_2)
x_train_3 = x_speech_train[0:len(x_train_1),speech_best_index_1]
x_train_3 = np.vstack(x_train_3)
x_train_4 = x_speech_train[0:len(x_train_1),speech_best_index_2]
x_train_4 = np.vstack(x_train_4)
x_train = np.concatenate((x_train_1,x_train_2,x_train_3,x_train_4), axis = -1)
x_test_1 = x_GSR_test[:,GSR_best_index_1]
x_test_1 = np.vstack(x_test_1)
x_test_2 = x_GSR_test[:,GSR_best_index_2]
x_test_2 = np.vstack(x_test_2)
x_test_3 = x_speech_test[0:len(x_test_1),speech_best_index_1]
x_test_3 = np.vstack(x_test_3)
x_test_4 = x_speech_test[0:len(x_test_1),speech_best_index_2]
x_test_4 = np.vstack(x_test_4)
x_test = np.concatenate((x_test_1,x_test_2,x_test_3,x_test_4),axis = -1)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train,y_GSR_train)
y_pred = clf.predict(x_test)
Score_f1 = f1_score(y_GSR_test,y_pred)
Score_recall = recall_score(y_GSR_test,y_pred)
Score_precision = precision_score(y_GSR_test,y_pred)
Score_accuracy = accuracy_score(y_GSR_test,y_pred)
print("F1 Score = ", Score_f1)
print("Recall = ", Score_recall)
print("Precision = ", Score_precision)
print("Accuracy = ", Score_accuracy)
