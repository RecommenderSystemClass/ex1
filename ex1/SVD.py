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

# P = np.random.random((users.size,K))
# Q = np.random.random((K, products.size))
P = []
for i in range(len(users)):
    P.append([random.random() for _ in range(K)])  # todo initialize P and Q with values [-1.5,1.5] and not just random

Q = []
for i in range(len(products)):
    Q.append(np.random.rand(K))

print("P[" + str(len(P)) + "]Q[" + str(len(P)) + "]")

lam = 0.02  # regularization #todo: learn this value X validations
delta = 0.005  # learning rate #todo: learn this value
mu = trainDataDF['stars'].mean()
bi = random.random()
bu = random.random()


def R(u, i):
    print(P[u] * Q[i] + "  " + P[u] + "  " + Q[i])
    return mu + bi + bu + P[u] * Q[i]


def E(u, i):
    return R(u, i)


def indexOfUser(user):
    return lUsers.index(user)


def indexOfProduct(product):
    return lProducts.index(product)


def handleRatingLine(user_id, business_id, stars, idx):
    global bu
    global bi
    u = indexOfUser(user_id)
    i = indexOfProduct(business_id)
    Rui = R(u, i)
    # Eui = Rui - stars
    # bu = bu + delta * (Eui - lam * bu)
    # bi = bi + delta * (Eui - lam * bi)
    # Q[i] = Q[i] + delta * (Eui * P(u) - lam * Q[i])
    # P[u] = P[u] + delta * (Eui * Q(i) - lam * P[u])
    print(user_id + " " + business_id + " " + str(stars) + " " + str(idx))


trainDataDF.apply(lambda x: handleRatingLine(x['user_id'], x['business_id'], x['stars'], x.name), axis=1)

exit(0)
