# Alex Daniels
# Gil Shamay 033076324

# requirments pythin3 x64, pandas, numpy
from builtins import print

from RMSE import *
from load import *
import pandas as pd
import numpy as np
import random

seed = 80
random.seed(seed)

trainData = './data/trainData.csv'
trainData = "D:/BGU/RS/EXs/ex1/ex1/data/trainData.csv"  # used this line for console debug
trainDataDF = load(trainData)

users = trainDataDF['user_id'].unique()
products = trainDataDF['business_id'].unique()
lUsers = list(users)
lProducts = list(products)
K = 100  # todo check values 100-500
lam = 0.02  # regularization #todo: learn this value X validations
delta = 0.005  # learning rate #todo: learn this value
mu = trainDataDF['stars'].mean()
bi = random.random()
bu = random.random()

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

print("begin values"
      + "bi[" + str(bi)
      + "]bu[" + str(bu)
      + "]K[" + str(K)
      + "]lambda[" + str(lam)
      + "]delta[" + str(delta)
      + "]mu[" + str(mu)
      + "]P len[" + str(len(P))
      + "]Q len [" + str(len(Q))
      + "]")


###########################################################

def R(u, i):
    # print(str(u) + " " + str(i) + "  " + str(P[u]) + "  " + str(Q[i]))
    return mu + bi + bu + P[u].dot(Q[i])


def E(u, i):
    return R(u, i)


def indexOfUser(user):
    return lUsers.index(user)


def indexOfProduct(product):
    return lProducts.index(product)

handleRatingLineCounter = 0
def handleRatingLine(user_id, business_id, stars, idx):
    global bu
    global bi
    global handleRatingLineCounter
    handleRatingLineCounter +=1
    u = indexOfUser(user_id)
    i = indexOfProduct(business_id)
    Rui = R(u, i)
    currentCalculatedRates.append(Rui)
    Eui = Rui - stars
    bu = bu + delta * (Eui - lam * bu)
    bi = bi + delta * (Eui - lam * bi)
    Q[i] = Q[i] + delta * (Eui * P[u] - lam * Q[i])
    P[u] = P[u] + delta * (Eui * Q[i] - lam * P[u])
    # print(user_id + " " + business_id + " " + str(stars) + " " + str(idx))


actualRates = trainDataDF['stars']
iterations = 0
currentRMSE = 0
lastRMSE = 0

while currentRMSE <= lastRMSE:
    # todo: keep last P and Q and use them -->keep the one with better error rate
    iterations += 1
    currentCalculatedRates = []
    trainDataDF.apply(lambda x: handleRatingLine(x['user_id'], x['business_id'], x['stars'], x.name), axis=1)

    lastRMSE = currentRMSE
    currentRMSE = RMSE(currentCalculatedRates, actualRates)
    print("lastRMSE [" + str(lastRMSE)
          + "]currentRMSE[" + str(currentRMSE)
          + "]iterations[" + str(iterations)
          + "]bi[" + str(bi)
          + "]bu[" + str(bu)
          + "]K[" + str(K)
          + "]lambda[" + str(lam)
          + "]delta[" + str(delta)
          + "]mu[" + str(mu) + "]")

print(" ---- Done ---- ")

exit(0)
