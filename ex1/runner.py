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
from SVD import *
import numpy as np
import random
import time
import sys
import math
import pandas as pd
import pickle
import socket

seed = 80
random.seed(seed)
#################################
# Parameters
SVDppOptions = [False]  # SVD++
Ks = [100]  # 400 # [100, 200, 300, 400, 500]
deltas = [0.05]  # 0.03 # [0.03, 0.04, 0.05, 0.06, 0.07]  # learning rate
lams = [0.05]  # 0.07 # [0.03, 0.04, 0.05, 0.06, 0.07]  # regularization
errorCalculation = 'RMSE'
# errorCalculation = 'MAE'

BInitialValuePlusMinusIntervals = [0.01]  # [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1]
PQInitialValuePlusMinusIntervals = [0.01]  # [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1]
YInitialValueInterval = 0.01

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
#   split train data to train and validations - take 30% of the users
#   then select 30% of the samples of those users to be the train data and all teh rest is teh test data
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
# run
for SVDpp in SVDppOptions:
    for K in Ks:
        for PQInitialValuePlusMinusInterval in PQInitialValuePlusMinusIntervals:
            for BInitialValuePlusMinusInterval in BInitialValuePlusMinusIntervals:
                for lam in lams:
                    lam2 = lam  # used for SVD++; can get a different value then lam
                    for delta in deltas:
                        #####################################
                        # Build Model
                        #####################################
                        ## SVD and SVD++ ##
                        mySvd = SVD(trainDataDF,
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
                                    )
                        ### sentiment ##
                        ### ensamble ##
                        ########   Predict on Test   ########
                        predictBeginTime = time.time()
                        calculatedTestRates = mySvd.predictRates(testDataDF)
                        PredictTime = time.time() - predictBeginTime
                        ######## Calc Error  ########
                        actuaTestRaes = testDataDF['stars'].to_list()
                        testRMSE = RMSE(actuaTestRaes, calculatedTestRates)
                        testMAE = MAE(actuaTestRaes, calculatedTestRates)
                        ########Print results and save log to file  ########
                        printDebug("********************************************************")
                        strFinalResult = ("SVD "
                                          + errorCalculation + "OnTrain[" + str(mySvd.lastError) + "]"
                                          + "RMSEOnTest[" + str(testRMSE) + "]"
                                          + "MAEOnTest[" + str(testMAE) + "]"
                                          + "K[" + str(K) + "]"
                                          + "lambda[" + str(lam) + "]"
                                          + "delta[" + str(delta) + "]"
                                          + "mu[" + str(mySvd.mu) + "]"
                                          + "learningTime[" + str(mySvd.learningTime) + "]"
                                          + "PredictTime[" + str(PredictTime) + "]"
                                          + "PQInitialInterval[+-" + str(PQInitialValuePlusMinusInterval) + "]"
                                          + "BInitialInterval[+-" + str(BInitialValuePlusMinusInterval) + "]"
                                          + "users[" + str(len(mySvd.P)) + "]"
                                          + "trainAndValidaiton Size[" + str(len(trainDataDF_all)) + "]"
                                          + "train Size[" + str(len(trainDataDF)) + "]"
                                          + "validation Size[" + str(len(validationDataDF)) + "]"
                                          + "test Orig Size[" + str(len(testDataDF_orig)) + "]"
                                          + "Test Size[" + str(len(testDataDF)) + "]"
                                          + "missingUsers[" + str(len(missingUsers)) + "]"
                                          + "missingProducts[" + str(len(missingProducts)) + "]"
                                          + "learn iterations[" + str(mySvd.iterations) + "]"
                                          + "SVDpp[" + str(SVDpp) + "]"
                                          + "YInitialValuePlusMinusInterval[" + str(YInitialValueInterval) + "]"
                                          + 'host[' + socket.gethostname() + "]"
                                          )
                        printDebug(strFinalResult)
                        printDebug("********************************************************")
                        fileBaseName = errorCalculation + "Train[" + str("%.5f" % mySvd.lastError) + "]" \
                                       + "RMSEOnTest[" + str(testRMSE) + "]" \
                                       + "MAEOnTest[" + str(testMAE) + "]" \
                                       + "K[" + str(K) + "]" \
                                       + "lambda[" + str(lam) + "]" \
                                       + "delta[" + str(delta) + "]" \
                                       + "learnSec[" + str("%.2f" % mySvd.learningTime) + "]" \
                                       + "PredictSec[" + str("%.2f" % PredictTime) + "]" \
                                       + "PQ[" + str(PQInitialValuePlusMinusInterval) + "]" \
                                       + "B[" + str(BInitialValuePlusMinusInterval) + "]" \
                                       + "SVDpp[" + str(SVDpp) + "]" \
                                       + "Y[" + str(YInitialValueInterval) + "]" \
                                       + 'host[' + socket.gethostname() + "]"
                        printToFile(".\\results\\" + fileBaseName + ".log")
                        ##########################################
                        # Save the Model to be reused
                        # dumpFileFullPath = ".\\dumps\\" + fileBaseName + ".dump"
                        # with open(dumpFileFullPath, 'wb') as fp:
                        #     pickle.dump(mySvd, fp)
                        ##########################################
                        # Load the Model to be reused
                        # mySvdload = None
                        # with open(dumpFileFullPath, 'rb') as fp:
                        #     mySvdload = pickle.load(fp)

printDebug(" ---- Done ---- ")
exit(0)
