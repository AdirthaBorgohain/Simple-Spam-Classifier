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

mails = pd.read_csv('spam.csv', encoding = 'latin-1')
X = mails.iloc[:, 1].values
y = mails.iloc[:, 0].values
dx = pd.DataFrame(X)
dy = pd.DataFrame(y)
