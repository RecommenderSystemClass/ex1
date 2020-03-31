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
lUsers = list(users)
lProducts = list(products)
K = 100  # todo check values 100-500
lam = 0.05  # regularization #todo: learn this value in X validations
delta = 0.05  # learning rate #todo: learn this value
mu = trainDataDF['stars'].mean()

# Bu and Bi with values  random in [-0.25,0.25]
Bu = np.random.rand(len(lUsers))
Bu = Bu * 0.5 - 0.25
Bi = np.random.rand(len(lProducts))
Bi = Bi * 0.5 - 0.25

# P and Q with values  random in [-0.25,0.25]
# P = np.random.random((users.size,K))
# Q = np.random.random((K, products.size))
P = []
for i in range(len(users)):
    P.append(np.random.rand(K) * 0.5 - 0.25)

Q = []
for i in range(len(products)):
    Q.append(np.random.rand(K) * 0.5 - 0.25)


###########################################################
def E(u, i):
    return R(u, i)


def indexOfUser(user):
    return lUsers.index(user)


def indexOfProduct(product):
    return lProducts.index(product)


def R(u, i):
    # print(str(u) + " " + str(i) + "  " + str(P[u]) + "  " + str(Q[i]))
    return mu + Bi[i] + Bu[u] + P[u].dot(Q[i])


handleRatingLineCounter = 0


def handleRatingLine(user_id, business_id, stars, idx):
    # print("handleRatingLine user_id[" + user_id + "]business_id[" + business_id + "]stars[" + str(stars) + "]idx[" + str(idx) + "]u[" + str(u) + "]i[" + str(i) +"]")
    global handleRatingLineCounter
    handleRatingLineCounter += 1
    u = indexOfUser(user_id)
    i = indexOfProduct(business_id)
    Rui = R(u, i)
    currentCalculatedRates.append(Rui)
    Eui = stars - Rui
    newQi = Q[i] + delta * (Eui * P[u] - lam * Q[i])
    newPu = P[u] + delta * (Eui * Q[i] - lam * P[u])
    newBu = Bu[u] + delta * (Eui - lam * Bu[u])
    newBi = Bi[i] + delta * (Eui - lam * Bi[i])
    ######################################
    # Test code
    # if(math.isnan(Rui)):
    #     print("error")
    #     return
    # if(math.isnan(Eui)):
    #     print("error")
    #     return
    # sumTestQ = np.sum(newQi)
    # sumTestP = np.sum(newPu)
    # if (np.isnan(sumTestQ) or np.isnan(sumTestP) or math.isinf(sumTestQ) or math.isinf(sumTestP)):
    #     print("error")
    #     return
    # if(math.fabs(newBu) > 10000 or math.fabs(newBi) > 10000 ):
    #     print("error")
    #     return
    ######################################
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
      + "avgBi[" + str(np.average(Bi)) + "]"
      + "avgBu[" + str(np.average(Bu)) + "]"
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
    handleRatingLineCounter = 0
    currentCalculatedRates = []
    # keep last P and Q and use them -->keep the one with better error rate
    lastP = P
    lastQ = Q
    trainDataDF.apply(lambda x: handleRatingLine(x['user_id'], x['business_id'], x['stars'], x.name), axis=1)
    lastRMSE = currentRMSE
    # todo: run Predict on the validation set and get RMSE
    # todo: add additional methion, other then RMSE
    currentRMSE = RMSE(currentCalculatedRates, actualRates)
    print("lastRMSE [" + str(lastRMSE) + "]"
          + "currentRMSE[" + str(currentRMSE) + "]"
          + "iterations[" + str(iterations) + "]"
          # + "bi[" + str(str(np.average(Bi))) + "]"
          # + "bu[" + str(str(np.average(Bu))) + "]"
          + "SecIter[" + str(time.time() - iterationBeginTime) + "]"
          + "SecBegin[" + str(time.time() - beginTime) + "]"
          )

# todo: use lastP and lastQ to predictRating

# todo: load test data
# todo: run predict on test and calc RMSE
print(" ---- Done ---- ")

exit(0)
