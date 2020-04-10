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
#################################
# Parameters

SVDppOptions = [False]  # SVD++
Ks = [400]  # [100, 200, 300, 400, 500]
deltas = [0.03]  # [0.03, 0.04, 0.05, 0.06, 0.07]  # learning rate
lams = [0.07]  # [0.03, 0.04, 0.05, 0.06, 0.07]  # regularization
BInitialValuePlusMinusIntervals = [0.01]  # [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1]
PQInitialValuePlusMinusIntervals = [0.01]  # [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1]
YInitialValueInterval = 0.01
errorCalculation = 'RMSE'
#errorCalculation = 'MAE'
#################################
#setting the error Method

errorMethod = None
if (errorCalculation == 'RMSE'):
    errorMethod = RMSE
else:
    errorMethod = MAE

#################################
# The Data
trainData = "D:/BGU/RS/EXs/ex1/ex1/data/trainData.csv"  # used this line for console debug
testData = "D:/BGU/RS/EXs/ex1/ex1/data/testData.csv"  # used this line for console debug
trainData = './data/trainData.csv'
testData = './data/testData.csv'

################################
# Load Data

trainDataDF_all = load(trainData)
################################
# Clean Data

len1 = len(trainDataDF_all)
trainDataDF_all.isnull().values.any()  # check validity of the data
trainDataDF_all = trainDataDF_all.dropna()
len2 = len(trainDataDF_all)
print("removed [" + str(len2 - len1) + "]null data entries")

################################
# Split train / validation

# split train data to train and validations - take 30% of the users
# then select 30% of the samples of those users to be the train data and all teh rest is teh test data
products = trainDataDF_all['business_id'].unique()
users = trainDataDF_all['user_id'].unique()
np.random.shuffle(users)
userSplit = int(len(users) * 0.7)
train_users, validation_users = users[:userSplit], users[userSplit:]
validationUsersData = trainDataDF_all.loc[trainDataDF_all['user_id'].isin(validation_users)]
validationDataDF = validationUsersData.sample(
    frac=0.3,  # take 30% of the data of the usres we selected for training
    random_state=seed)  # we take here 30% of the validaitons users ratings randomally
trainDataDF = trainDataDF_all.drop(validationDataDF.index)
trainDataDF = trainDataDF.sample(frac=1).reset_index(drop=True)  # shuffle train Data #todo: Check if really needed

trainProducts = trainDataDF['business_id'].unique()
trainUsers = trainDataDF['user_id'].unique()

################################
# load test data
testDataDF_orig = load(testData)

################################
# clean test and validaiton data - remove all entries with new users or product (no cold start)

validationDataDF = validationDataDF.loc[validationDataDF['user_id'].isin(trainUsers)]
validationDataDF = validationDataDF.loc[validationDataDF['business_id'].isin(trainProducts)]

testDataDF = testDataDF_orig.loc[testDataDF_orig['user_id'].isin(trainUsers)]
testDataDF = testDataDF.loc[testDataDF['business_id'].isin(trainProducts)]
testDataFilteredOut = testDataDF_orig.drop(testDataDF.index)
missingProducts = testDataFilteredOut['business_id'].unique()
missingUsers = testDataFilteredOut['user_id'].unique()
################################


for SVDpp in SVDppOptions:
    for K in Ks:
        for PQInitialValuePlusMinusInterval in PQInitialValuePlusMinusIntervals:
            for BInitialValuePlusMinusInterval in BInitialValuePlusMinusIntervals:
                for lam in lams:
                    lam2 = lam  # used for SVD++; can get a different value then lam
                    for delta in deltas:
                        mu = trainDataDF['stars'].mean()
                        t1 = time.time()
                        Bu = {}
                        Bi = {}
                        Yu = {}
                        RuMi1_2 = {}
                        P = {}
                        usersNumberOfRates = trainDataDF['user_id'].value_counts()
                        usersNumberOfRatesDic = usersNumberOfRates.to_dict()  # calculate this once - it is used many times

                        for user in trainUsers:
                            P[user] = (np.random.rand(K) * (
                                    PQInitialValuePlusMinusInterval * 2) - PQInitialValuePlusMinusInterval)
                            Bu[user] = np.random.rand() * (
                                    BInitialValuePlusMinusInterval * 2) - BInitialValuePlusMinusInterval
                            if (SVDpp):
                                numOfUserRates = usersNumberOfRatesDic[user]
                                Yu[user] = np.random.rand(numOfUserRates, K) * (
                                        YInitialValueInterval * 2) - YInitialValueInterval
                                RuMi1_2[user] = pow(numOfUserRates, -0.5)

                        printDebug("users init took[" + str(time.time() - t1) + "]")
                        Q = {}
                        t1 = time.time()
                        for product in products:
                            Q[product] = (
                                    np.random.rand(K) * (
                                    PQInitialValuePlusMinusInterval * 2) - PQInitialValuePlusMinusInterval)
                            Bi[product] = np.random.rand() * (
                                    BInitialValuePlusMinusInterval * 2) - BInitialValuePlusMinusInterval
                        printDebug("items init took[" + str(time.time() - t1) + "]")


                        ###########################################################

                        def calculateSingleRate(ratingLine):
                            u = ratingLine['user_id']
                            i = ratingLine['business_id']
                            if (not SVDpp):
                                return mu + Bi[i] + Bu[u] + P[u].dot(Q[i])
                            else:
                                RuMi1_2u = RuMi1_2[u]
                                y = Yu[u]
                                sigmaYu = np.sum(y, axis=0)
                                return mu + Bi[i] + Bu[u] + Q[i].dot(P[u] + RuMi1_2u * sigmaYu)


                        def handleRatingLine(ratingLine):
                            user_id = ratingLine['user_id']
                            business_id = ratingLine['business_id']
                            stars = ratingLine['stars']
                            q = Q[business_id]
                            p = P[user_id]
                            bu = Bu[user_id]
                            bi = Bi[business_id]
                            Rui = None
                            y = None
                            RuMi1_2u = None
                            sigmaYu = None
                            if (not SVDpp):
                                Rui = mu + bi + bu + p.dot(q)
                            else:
                                y = Yu[user]
                                RuMi1_2u = RuMi1_2[user]
                                sigmaYu = np.sum(y, axis=0)
                                Rui = mu + bi + bu + q.dot(p + RuMi1_2u * sigmaYu)

                            Eui = stars - Rui
                            Bu[user_id] = bu + delta * (Eui - lam * bu)
                            Bi[business_id] = bi + delta * (Eui - lam * bi)
                            if (not SVDpp):
                                Q[business_id] = q + delta * (Eui * p - lam * q)
                                P[user_id] = p + delta * (Eui * q - lam * p)
                            else:
                                Q[business_id] = q + delta * (Eui * (p + RuMi1_2u * sigmaYu) - lam2 * q)
                                P[user_id] = p + delta * (Eui * q - lam2 * p)
                                ERuMi1_2u = Eui * RuMi1_2u
                                Yu[user] = [Yj + delta * (ERuMi1_2u * q - lam2 * Yj) for Yj in y]


                        actualRates = validationDataDF['stars'].to_list()
                        iterations = 0
                        currentError = sys.maxsize - 1
                        lastError = sys.maxsize

                        beginTime = time.time()
                        printDebug("Beggining: "
                                   + "K[" + str(K) + "]"
                                   + "lambda[" + str(lam) + "]"
                                   + "delta[" + str(delta) + "]"
                                   + "PQInitialInterval[+-" + str(PQInitialValuePlusMinusInterval) + "]"
                                   + "BInitialInterval[+-" + str(BInitialValuePlusMinusInterval) + "]"
                                   + "mu[" + str(mu) + "]"
                                   + "P len[" + str(len(P)) + "]"
                                   + "Q len [" + str(len(Q)) + "]"
                                   + "SVDpp[" + str(SVDpp) + "]"
                                   )

                        lastP = []
                        lastQ = []


                        def predictRates(newDataFrame):
                            return newDataFrame.apply(calculateSingleRate, axis=1).tolist()


                        deltaOrig = delta
                        while currentError < lastError:
                            iterationBeginTime = time.time()
                            iterations += 1
                            # keep last P and Q and use them -->keep the one with better error rate
                            lastP = P
                            lastQ = Q
                            trainDataDF.apply(handleRatingLine, axis=1)
                            if (SVDpp):
                                delta = delta * 0.9
                            currentCalculatedRates = predictRates(validationDataDF)
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

                        delta = deltaOrig
                        P = lastP
                        Q = lastQ
                        learningTime = time.time() - beginTime
                        ########   End Of Model Build   ########

                        ########   Predict on Test   ########
                        predictBeginTime = time.time()
                        calculatedTestRates = predictRates(testDataDF)
                        PredictTime = time.time() - predictBeginTime
                        actuaTestRaes = testDataDF['stars'].to_list()

                        testRMSE = RMSE(actuaTestRaes, calculatedTestRates)
                        testMAE = MAE(actuaTestRaes, calculatedTestRates)

                        printDebug("********************************************************")
                        strFinalResult = ("SVD "
                                          + errorCalculation + "OnTrain[" + str(lastError) + "]"
                                          + "RMSEOnTest[" + str(testRMSE) + "]"
                                          + "MAEOnTest[" + str(testMAE) + "]"
                                          + "K[" + str(K) + "]"
                                          + "lambda[" + str(lam) + "]"
                                          + "delta[" + str(delta) + "]"
                                          + "mu[" + str(mu) + "]"
                                          + "learningTime[" + str(learningTime) + "]"
                                          + "PredictTime[" + str(PredictTime) + "]"
                                          + "PQInitialInterval[+-" + str(PQInitialValuePlusMinusInterval) + "]"
                                          + "BInitialInterval[+-" + str(BInitialValuePlusMinusInterval) + "]"
                                          + "users[" + str(len(P)) + "]"
                                          + "trainAndValidaiton Size[" + str(len(trainDataDF_all)) + "]"
                                          + "train Size[" + str(len(trainDataDF)) + "]"
                                          + "validation Size[" + str(len(validationDataDF)) + "]"
                                          + "test Orig Size[" + str(len(testDataDF_orig)) + "]"
                                          + "Test Size[" + str(len(testDataDF)) + "]"
                                          + "missingUsers[" + str(len(missingUsers)) + "]"
                                          + "missingProducts[" + str(len(missingProducts)) + "]"
                                          + "learn iterations[" + str(iterations) + "]"
                                          + "SVDpp[" + str(SVDpp) + "]"
                                          + "YInitialValuePlusMinusInterval[" + str(YInitialValueInterval) + "]"
                                          + 'host[' + socket.gethostname() + "]"
                                          )
                        printDebug(strFinalResult)
                        printDebug("********************************************************")

                        fileBaseName = errorCalculation + "Train[" + str("%.5f" % lastError) + "]" \
                                       + "RMSEOnTest[" + str(testRMSE) + "]" \
                                       + "MAEOnTest[" + str(testMAE) + "]" \
                                       + "K[" + str(K) + "]" \
                                       + "lambda[" + str(lam) + "]" \
                                       + "delta[" + str(delta) + "]" \
                                       + "learnSec[" + str("%.2f" % learningTime) + "]" \
                                       + "PredictSec[" + str("%.2f" % PredictTime) + "]" \
                                       + "PQ[" + str(PQInitialValuePlusMinusInterval) + "]" \
                                       + "B[" + str(BInitialValuePlusMinusInterval) + "]" \
                                       + "SVDpp[" + str(SVDpp) + "]" \
                                       + "Y[" + str(YInitialValueInterval) + "]" \
                                       + 'host[' + socket.gethostname() + "]"

                        printToFile(".\\results\\" + fileBaseName + ".log")

                        # class mySvd:
                        #     def __init__(mysillyobject,
                        #                  p,
                        #                  q,
                        #                  Bu,
                        #                  Bi,
                        #                  mu,
                        #                  rmse,
                        #                  Yu):
                        #         mysillyobject.p = p
                        #         mysillyobject.q = q
                        #         mysillyobject.Bi = Bi
                        #         mysillyobject.Bu = Bu
                        #         mysillyobject.mu = mu
                        #         mysillyobject.rmse = rmse
                        #         mysillyobject.Yu = Yu

                        # mySvdSave = mySvd(P, Q, Bu, Bi, mu, lastError, Yu)
                        # with open(filePath + ".dump", 'wb') as fp:
                        #     pickle.dump(mySvdSave, fp)

                        # mySvdload = None
                        # with open(filePath, 'rb') as fp:
                        #     mySvdload = pickle.load(fp)


class SVD:
    def __init__(this, TrainingDataFrame, parameter, SVDpp, ErrorMethod):
        this.SVDpp = SVDpp

    def PredictRating(this):
        printDebug("PredictRating++")


printDebug(" ---- Done ---- ")
exit(0)
