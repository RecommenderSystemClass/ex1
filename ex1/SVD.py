# Alex Danieli 317618718
# Gil Shamay 033076324

# requirments pythin3 x64, pandas, numpy
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

trainDataDF.isnull().values.any()  # todo: check validity of teh data

users = trainDataDF['user_id'].unique()
products = trainDataDF['business_id'].unique()
lUsers = list(users)
lProducts = list(products)
K = 100  # todo check values 100-500
lam = 0.02  # regularization #todo: learn this value X validations
delta = 0.05  # learning rate #todo: learn this value
mu = trainDataDF['stars'].mean()
Bu = np.random.rand(len(lUsers))
Bi = np.random.rand(len(lProducts))

# todo initialize P and Q with values [-1.5,1.5] and not just random
# P = np.random.random((users.size,K))
# Q = np.random.random((K, products.size))
P = []
for i in range(len(users)):
    # P.append([random.random() for _ in range(K)])
    P.append(np.random.rand(K))

Q = []
for i in range(len(products)):
    # Q.append([random.random() for _ in range(K)])
    Q.append(np.random.rand(K))


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
    global handleRatingLineCounter
    handleRatingLineCounter += 1
    u = indexOfUser(user_id)
    i = indexOfProduct(business_id)
    Rui = R(u, i)
    if(math.isnan(Rui)):
        print("error")
        return
    currentCalculatedRates.append(Rui)
    Eui = stars - Rui
    if(math.isnan(Eui)):
        print("error")
        return
    Bu[u] = Bu[u] + delta * (Eui - lam * Bu[u])
    Bi[i] = Bi[i] + delta * (Eui - lam * Bi[i])
    # print("handleRatingLine user_id[" + user_id + "]business_id[" + business_id + "]stars[" + str(stars) + "]idx[" + str(idx) + "]u[" + str(u) + "]i[" + str(i) +"]")
    newQi = Q[i] + delta * (Eui * P[u] - lam * Q[i])
    newPu = P[u] + delta * (Eui * Q[i] - lam * P[u])
    if (np.isnan(np.sum(newQi)) or np.isnan(np.sum(newPu))):
        print("error")
        return
    P[u] = newPu
    Q[i] = newQi


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

while currentRMSE < lastRMSE:
    iterationBeginTime = time.time()
    # todo: keep last P and Q and use them -->keep the one with better error rate
    iterations += 1
    currentCalculatedRates = []
    trainDataDF.apply(lambda x: handleRatingLine(x['user_id'], x['business_id'], x['stars'], x.name), axis=1)

    lastRMSE = currentRMSE
    currentRMSE = RMSE(currentCalculatedRates, actualRates)
    print("lastRMSE [" + str(lastRMSE) + "]"
          + "currentRMSE[" + str(currentRMSE) + "]"
          + "iterations[" + str(iterations) + "]"
          + "bi[" + str(str(np.average(Bi))) + "]"
          + "bu[" + str(str(np.average(Bu))) + "]"
          + "SecIter[" + str(time.time() - iterationBeginTime) + "]"
          + "SecBegin[" + str(time.time() - beginTime) + "]"
          )

print(" ---- Done ---- ")

exit(0)
