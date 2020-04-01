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

seed = 80
random.seed(seed)

trainData = './data/trainData.csv'
# trainData = "D:/BGU/RS/EXs/ex1/ex1/data/trainData.csv"  # used this line for console debug
trainDataDF = load(trainData)
# todo: split train validations 30% of the users
# todo: split select validations lines to validate only on 30% of the products

trainDataDF = trainDataDF.sample(frac=1).reset_index(drop=True)  # shuffle train Data
trainDataDF.isnull().values.any()  # todo: check validity of the data

users = trainDataDF['user_id'].unique()
products = trainDataDF['business_id'].unique()

indexUsers = 0
dicUsers = {}
for user in users:
    dicUsers[user] = indexUsers
    indexUsers += 1

indexProducts = 0
dicProducts = {}
for product in products:
    dicProducts[product] = indexProducts
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
    return R(u, i)


def R(u, i):
    return mu + Bi[i] + Bu[u] + P[u].dot(Q[i])


def handleRatingLine(user_id, business_id, stars, idx):
    u = user_id
    i = business_id
    q = Q[i]
    p = P[u]
    bu = Bu[u]
    bi = Bi[i]
    Rui = mu + bi + bu + p.dot(q)
    currentCalculatedRates.append(
        Rui)  # todo: this will not be needed when calculating RMSE will be calc on validaiton set
    Eui = stars - Rui
    newQi = q + delta * (Eui * p - lam * q)
    newPu = p + delta * (Eui * q - lam * p)
    newBu = bu + delta * (Eui - lam * bu)
    newBi = bi + delta * (Eui - lam * bi)
    Q[i] = newQi
    P[u] = newPu
    Bu[u] = newBu
    Bi[i] = newBi


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
    trainDataDF.apply(lambda x: handleRatingLine(x['user_id'], x['business_id'], x['stars'], x.name), axis=1)
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
