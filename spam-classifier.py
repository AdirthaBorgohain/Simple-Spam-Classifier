#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 23:57:04 2018

@author: adirtha
"""
 # for processing messages 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
# for visualization
import matplotlib.pyplot as plt
from wordcloud import WordCloud
# for calculations
from math import log,sqrt
# for loading data
import pandas as pd
# for generating test-train split
import numpy as np

# read csv file
mails = pd.read_csv('spam.csv', encoding = 'latin-1')

# remove unneeded columns
mails.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace = True)

# rename the two columns
mails.rename(columns = {'v1': 'labels', 'v2': 'message'}, inplace = True)

# map ham and spam to 0 and 1 respectively
mails['label'] = mails['labels'].map({'ham': 0, 'spam': 1})

# remove 'labels' column
mails.drop(['labels'], axis = 1, inplace = True)

totalMails = 4825 + 747
trainIndex, testIndex = list(), list()

for i in range(mails.shape[0]):
    if np.random.uniform(0, 1) < 0.75:
        trainIndex += [i]
    else:
        testIndex += [i]
trainData = mails.loc[trainIndex]
testData = mails.loc[testIndex]

trainData.reset_index(inplace = True)
trainData.drop(['index'], axis = 1, inplace = True)
testData.reset_index(inplace = True)
testData.drop(['index'], axis = 1, inplace = True)