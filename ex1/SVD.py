# Alex Danieli 317618718
# Gil Shamay 033076324

# requirements:
#   python3 x64,
#   pandas,
#   numpy,
#   surprise

from builtins import print
from RMSE import *
from MAE import *
from printDebug import *
from load import *
import numpy as np
import random
import sys
import time
import math
import pandas as pd
import pickle
import socket

seed = 80
random.seed(seed)

# accuracy - a variable to reduce runtime (for a trade off between runtime and accuracy)
#   if accuracy is > 0
#   for example -  use accuracy of 10000 for 4 digits accuracy results 0.xxxx
#   runtime can be very long in some cases if we don't use it and stop the algorithm in some acceptable accuracy
accuracy = 0  # 10000


#################################

class SVD:
    #############################################
    def predictRates(this, newDataFrame):
        return newDataFrame.apply(this.calculateSingleRate, axis=1).tolist()

    #############################################
    def calculateSingleRate(this, ratingLine):
        u = ratingLine['user_id']
        i = ratingLine['business_id']
        if (not this.SVDpp):
            return this.mu + this.Bi[i] + this.Bu[u] + this.P[u].dot(this.Q[i])
        else:
            RuMi1_2u = this.RuMi1_2[u]
            y = this.Yu[u]
            sigmaYu = np.sum(y, axis=0)
            return this.mu + this.Bi[i] + this.Bu[u] + this.Q[i].dot(this.P[u] + RuMi1_2u * sigmaYu)

    def __init__(this,
                 trainDataDF,
                 validationDataDF,
                 K,  # = 400
                 lam,  # = 0.07
                 delta,  # = 0.03
                 errorCalculation,  # = 'RMSE' / 'MAE'
                 PQInitialValuePlusMinusInterval,  # = 0.01
                 BInitialValuePlusMinusInterval,  # = 0.01
                 SVDpp,  # = False/True
                 lam2,  # = 0.07
                 YInitialValueInterval  # = 0.01
                 ):

        #################################
        # setting the error Method
        errorMethod = None
        if (errorCalculation == 'RMSE'):
            errorMethod = RMSE
        else:
            errorMethod = MAE

        this.SVDpp = SVDpp
        this.K = K
        this.lam = lam
        this.delta = delta
        this.errorCalculation = errorCalculation
        this.PQInitialValuePlusMinusInterval = PQInitialValuePlusMinusInterval
        this.BInitialValuePlusMinusInterval = PQInitialValuePlusMinusInterval
        this.SVDpp = SVDpp
        this.lam2 = SVDpp
        this.YInitialValueInterval = YInitialValueInterval
        trainDataDF.drop(['text'], axis=1, inplace=True)
        trainUsers = trainDataDF['user_id'].unique()
        products = trainDataDF['business_id'].unique()

        this.mu = trainDataDF['stars'].mean()
        this.Bu = {}
        this.Bi = {}
        this.Yu = {}
        this.RuMi1_2 = {}
        this.P = {}
        this.Q = {}
        usersNumberOfRates = trainDataDF['user_id'].value_counts()
        usersNumberOfRatesDic = usersNumberOfRates.to_dict()  # calculate this once - it is used many times

        t1 = time.time()
        for user in trainUsers:
            this.P[user] = (np.random.rand(K) * (
                    PQInitialValuePlusMinusInterval * 2) - PQInitialValuePlusMinusInterval)
            this.Bu[user] = np.random.rand() * (
                    BInitialValuePlusMinusInterval * 2) - BInitialValuePlusMinusInterval
            if (SVDpp):
                numOfUserRates = usersNumberOfRatesDic[user]
                this.Yu[user] = np.random.rand(numOfUserRates, K) * (
                        YInitialValueInterval * 2) - YInitialValueInterval
                this.RuMi1_2[user] = pow(numOfUserRates, -0.5)

        printDebug("users init took[" + str(time.time() - t1) + "]")

        t1 = time.time()
        for product in products:
            this.Q[product] = (
                    np.random.rand(K) * (
                    PQInitialValuePlusMinusInterval * 2) - PQInitialValuePlusMinusInterval)
            this.Bi[product] = np.random.rand() * (
                    BInitialValuePlusMinusInterval * 2) - BInitialValuePlusMinusInterval
        printDebug("items init took[" + str(time.time() - t1) + "]")

        ###########################################################

        def handleRatingLine(ratingLine):
            user_id = ratingLine['user_id']
            business_id = ratingLine['business_id']
            stars = ratingLine['stars']
            q = this.Q[business_id]
            p = this.P[user_id]
            bu = this.Bu[user_id]
            bi = this.Bi[business_id]
            Rui = None
            y = None
            RuMi1_2u = None
            sigmaYu = None
            if (not SVDpp):
                Rui = this.mu + bi + bu + p.dot(q)
            else:
                y = this.Yu[user]
                RuMi1_2u = this.RuMi1_2[user]
                sigmaYu = np.sum(y, axis=0)
                Rui = this.mu + bi + bu + q.dot(p + RuMi1_2u * sigmaYu)

            Eui = stars - Rui
            this.Bu[user_id] = bu + delta * (Eui - lam * bu)
            this.Bi[business_id] = bi + delta * (Eui - lam * bi)
            if (not SVDpp):
                this.Q[business_id] = q + delta * (Eui * p - lam * q)
                this.P[user_id] = p + delta * (Eui * q - lam * p)
            else:
                this.Q[business_id] = q + delta * (Eui * (p + RuMi1_2u * sigmaYu) - lam2 * q)
                this.P[user_id] = p + delta * (Eui * q - lam2 * p)
                ERuMi1_2u = Eui * RuMi1_2u
                this.Yu[user] = [Yj + delta * (ERuMi1_2u * q - lam2 * Yj) for Yj in y]

        actualRates = validationDataDF['stars'].to_list()
        iterations = 0
        lastError = sys.maxsize
        if (accuracy > 0):
            lastError = sys.maxsize / (accuracy * 10)  # avoid overflow in the while
        currentError = lastError - 1

        beginTime = time.time()
        printDebug("Beggining: "
                   + "K[" + str(K) + "]"
                   + "lambda[" + str(lam) + "]"
                   + "delta[" + str(delta) + "]"
                   + "PQInitialInterval[+-" + str(PQInitialValuePlusMinusInterval) + "]"
                   + "BInitialInterval[+-" + str(BInitialValuePlusMinusInterval) + "]"
                   + "mu[" + str(this.mu) + "]"
                   + "P len[" + str(len(this.P)) + "]"
                   + "Q len [" + str(len(this.Q)) + "]"
                   + "SVDpp[" + str(SVDpp) + "]"
                   )

        lastP = []
        lastQ = []

        deltaOrig = delta
        while (accuracy > 0) and (int(currentError * accuracy) < int(lastError * accuracy)) or ((accuracy <= 0) and (currentError < lastError)):
            iterationBeginTime = time.time()
            iterations += 1
            # keep last P and Q and use them -->keep the one with better error rate
            lastP = this.P
            lastQ = this.Q
            trainDataDF.apply(handleRatingLine, axis=1)
            if (SVDpp):
                delta = delta * 0.9
            currentCalculatedRates = this.predictRates(validationDataDF)
            lastError = currentError
            currentError = errorMethod(currentCalculatedRates, actualRates)
            printDebug("last" + errorCalculation + "[" + str(lastError) + "]"
                       + "current" + errorCalculation + "[" + str(currentError) + "]"
                       + "iterations[" + str(iterations) + "]"
                       + "SecIter[" + str(time.time() - iterationBeginTime) + "]"
                       + "SecBegin[" + str(time.time() - beginTime) + "]"
                       + "SVDpp[" + str(SVDpp) + "]"
                       )
            ########   End Of learn   ########


        this.P = lastP
        this.Q = lastQ
        this.lastError = lastError
        this.iterations = iterations
        this.learningTime = time.time() - beginTime
        printDebug("########   End Of SVD Model Build   ########")
