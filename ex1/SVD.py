from RMSE import *
from load import *
import pandas as pd
import numpy as np
import random

seed = 80
random.seed(seed)

trainData = './data/trainData.csv'
trainDataDF = load(trainData)
# trainDataDF = load("D:/BGU/RS/EXs/ex1/data/trainData.csv") #used for console debug

users = trainDataDF['user_id'].unique()
products = trainDataDF['business_id'].unique()

K = 100000
P = np.random.random((users.size,K))#not working!
Q = np.random.random((K, products.size))

##