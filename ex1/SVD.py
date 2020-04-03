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
testData = "D:/BGU/RS/EXs/ex1/ex1/data/testData.csv"  # used this line for console debug
trainDataDF_all = load(trainData)
trainDataDF_all.isnull().values.any()  # check validity of the data

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


###########################################################
def E(u, i):
    return calculateSingleRate(u, i)


def calculateSingleRate(ratingLine):
    u = ratingLine['user_id']
    i = ratingLine['business_id']
    return mu + Bi[i] + Bu[u] + P[u].dot(Q[i])


def handleRatingLine(user_id, business_id, stars):
    q = Q[business_id]
    p = P[user_id]
    bu = Bu[user_id]
    bi = Bi[business_id]
    Rui = mu + bi + bu + p.dot(q)  # R(u, i)
    Eui = stars - Rui
    Q[business_id] = q + delta * (Eui * p - lam * q)
    P[user_id] = p + delta * (Eui * q - lam * p)
    Bu[user_id] = bu + delta * (Eui - lam * bu)
    Bi[business_id] = bi + delta * (Eui - lam * bi)


actualRates = validationDataDF['stars'].to_list()
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


def predictRates(newDataFrame):
    return newDataFrame.apply(calculateSingleRate, axis=1).tolist()


while currentRMSE < lastRMSE:
    iterationBeginTime = time.time()
    iterations += 1
    # keep last P and Q and use them -->keep the one with better error rate
    lastP = P
    lastQ = Q

    trainDataDF.apply(lambda x: handleRatingLine(x['user_id'], x['business_id'], x['stars']), axis=1)
    currentCalculatedRates = predictRates(validationDataDF)
    lastRMSE = currentRMSE
    # todo: add additional method, other then RMSE
    currentRMSE = RMSE(currentCalculatedRates, actualRates)
    print("lastRMSE [" + str(lastRMSE) + "]"
          + "currentRMSE[" + str(currentRMSE) + "]"
          + "iterations[" + str(iterations) + "]"
          + "SecIter[" + str(time.time() - iterationBeginTime) + "]"
          + "SecBegin[" + str(time.time() - beginTime) + "]"
          )

P = lastP
Q = lastQ


class mySvd:
    def __init__(mysillyobject,
                 p,
                 q,
                 Bu,
                 Bi,
                 mu):
        mysillyobject.p = p
        mysillyobject.q = q
        mysillyobject.Bi = Bi
        mysillyobject.Bu = Bu
        mysillyobject.mu = mu


mySvdSave = mySvd(P, Q, Bu, Bi, mu)
filePath = '.\data\K' + str(K) + '_lam' + str(lam) + '_delta' + str(delta) + '.dump'
with open(filePath, 'wb') as fp:
    pickle.dump(mySvdSave, fp)

mySvdload = None
with open(filePath, 'rb') as fp:
    mySvdload = pickle.load(fp)

testDataDF = load(testData)
# todo: clean test data - remove all entries with new users or product (no cold start)
calculatedTestRates = predictRates(testDataDF)
actuaTestRaes = testDataDF['stars'].to_list()
testRMSE = RMSE(actuaTestRaes, calculatedTestRates)
print("RMSE on test[" + str(testRMSE) + "]")

print(" ---- Done ---- ")
exit(0)
