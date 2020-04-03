# Alex Danieli 317618718
# Gil Shamay 033076324

# requirments python3 x64, pandas, numpy
from builtins import print

from RMSE import *
from load import *
import numpy as np
import random
import sys
import time
import math
import pandas as pd
import pickle

seed = 80
random.seed(seed)

# trainData = './data/trainData.csv'
trainData = "D:/BGU/RS/EXs/ex1/ex1/data/trainData.csv"  # used this line for console debug
trainDataDF_all = load(trainData)
trainDataDF_all.isnull().values.any()  # todo: check validity of the data

# split train data to train and validations - take 30% of the users
# then select 30% of the samples of those users to be the train data and all teh rest is teh test data
products = trainDataDF_all['business_id'].unique()
users = trainDataDF_all['user_id'].unique()
np.random.shuffle(users)
userSplit = int(len(users) * 0.7)
train_users, validation_users = users[:userSplit], users[userSplit:]
validationUsersData = trainDataDF_all.loc[trainDataDF_all['user_id'].isin(validation_users)]
validationDataDF = validationUsersData.sample(frac=0.3,
                                              random_state=seed)  # todo: Should we take here 30% of the product that the validations users use or 30% randomally  ?
trainDataDF = trainDataDF_all.drop(validationDataDF.index)
trainDataDF = trainDataDF.sample(frac=1).reset_index(drop=True)  # shuffle train Data #todo: Check if really needed

trainProducts = trainDataDF['business_id'].unique()
trainUsers = trainDataDF['user_id'].unique()

indexUsers = 0
trainUsersDic = {}
for user in trainUsers:
    trainUsersDic[user] = indexUsers
    indexUsers += 1

indexProducts = 0
trainProductsDic = {}
for product in trainProducts:
    trainProductsDic[product] = indexProducts
    indexProducts += 1

K = 100  # todo check values 100-500
lam = 0.05  # regularization #todo: learn this value in X validations
delta = 0.05  # learning rate #todo: learn this value
mu = trainDataDF['stars'].mean()
Bu = {}
Bi = {}
P = {}
for user in users:
    P[user] = (np.random.rand(K) * 0.5 - 0.25)
    Bu[user] = np.random.rand() * 0.5 - 0.25

Q = {}
for product in products:
    Q[product] = (np.random.rand(K) * 0.5 - 0.25)
    Bi[product] = np.random.rand() * 0.5 - 0.25

with open('.\data\P.dump', 'wb') as fp:
    pickle.dump(P, fp)

with open('.\data\P.dump', 'rb') as fp:
    PP = pickle.load(fp)


###########################################################
def E(u, i):
    return R(u, i)


def R(u, i):
    return mu + Bi[i] + Bu[u] + P[u].dot(Q[i])


def handleRatingLine(user_id, business_id, stars):
    q = Q[business_id]
    p = P[user_id]
    bu = Bu[user_id]
    bi = Bi[business_id]
    Rui = mu + bi + bu + p.dot(q)
    currentCalculatedRates.append(
        Rui)  # todo: this will not be needed when calculating RMSE will be calc on validaiton set
    Eui = stars - Rui
    Q[business_id] = q + delta * (Eui * p - lam * q)
    P[user_id] = p + delta * (Eui * q - lam * p)
    Bu[user_id] = bu + delta * (Eui - lam * bu)
    Bi[business_id] = bi + delta * (Eui - lam * bi)


actualRates = trainDataDF['stars'].to_list()
iterations = 0
currentRMSE = sys.maxsize - 1
lastRMSE = sys.maxsize

beginTime = time.time()
print("Beggining: "
      + "K[" + str(K) + "]"
      + "lambda[" + str(lam) + "]"
      + "delta[" + str(delta) + "]"
      + "mu[" + str(mu) + "]"
      + "P len[" + str(len(P)) + "]"
      + "Q len [" + str(len(Q)) + "]"
      )

lastP = []
lastQ = []
while currentRMSE < lastRMSE:
    iterationBeginTime = time.time()
    iterations += 1
    currentCalculatedRates = []
    # keep last P and Q and use them -->keep the one with better error rate
    lastP = P
    lastQ = Q

    trainDataDF.apply(lambda x: handleRatingLine(x['user_id'], x['business_id'], x['stars']), axis=1)
    lastRMSE = currentRMSE
    # todo: run Predict on the validation set and get RMSE
    # todo: add additional methion, other then RMSE
    # currentRMSE = RMSE(currentCalculatedRates, actualRates)
    print("lastRMSE [" + str(lastRMSE) + "]"
          + "currentRMSE[" + str(currentRMSE) + "]"
          + "iterations[" + str(iterations) + "]"
          + "SecIter[" + str(time.time() - iterationBeginTime) + "]"
          + "SecBegin[" + str(time.time() - beginTime) + "]"
          )

# todo: use lastP and lastQ to predictRating

# todo: load test data
# todo: run predict on test and calc RMSE
print(" ---- Done ---- ")

exit(0)
