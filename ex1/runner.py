# Alex Danieli 317618718
# Gil Shamay 033076324

# requirements:
#   python3 x64,
#   pandas,
#   numpy,
#   surprise,
#   nltk,
#   sklearn,
#   category-encoders


from builtins import print
from RMSE import *
from MAE import *
from printDebug import *
from load import *
from SVD import *
from RecommenderSystem import *
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
errorCalculation = 'RMSE'  # select the error calculation method used to build the SVD / SVD++ model with
# errorCalculation = 'MAE'
ensamble = True

# initial random values range for the different matrix
BInitialValuePlusMinusIntervals = [0.01]  # [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1]
PQInitialValuePlusMinusIntervals = [0.01]  # [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1]
YInitialValueInterval = 0.01

#################################
# The Data
trainData = './data/trainData.csv'
testData = './data/testData.csv'


################################
# Helpers
################################
def cleanDataFromNulls(data):
    len1 = len(data)
    data.isnull().values.any()  # check validity of the data
    data = data.dropna()
    len2 = len(data)
    print("removed [" + str(len2 - len1) + "]null data entries")
    return data


def splitTrainValidation(trainDataDF_all):
    #   split train data to train and validations - take 30% of the users
    #   then select 30% of the samples of those users to be the train data and all teh rest is teh test data
    users = trainDataDF_all['user_id'].unique()
    np.random.shuffle(users)
    userSplit = int(len(users) * 0.7)
    validation_users = users[userSplit:]
    validationUsersData = trainDataDF_all.loc[trainDataDF_all['user_id'].isin(validation_users)]
    validationDataDF = validationUsersData.sample(
        frac=0.3,  # take 30% of the data of the usres we selected for training
        random_state=seed)  # we take here 30% of the validaitons users ratings randomally
    trainDataDF = trainDataDF_all.drop(validationDataDF.index)
    trainDataDF = trainDataDF.sample(frac=1).reset_index(drop=True)  # shuffle train Data #todo: Check if really needed
    trainProducts = trainDataDF['business_id'].unique()
    trainUsers = trainDataDF['user_id'].unique()
    # Clean validations from cold start
    validationDataDF, missingProducts, missingUsers = cleanColdStarts(trainProducts, trainUsers, validationDataDF)
    return validationDataDF, trainDataDF, trainProducts, trainUsers


def cleanColdStarts(trainProducts, trainUsers, data_orig):
    # clean validaiton data - remove all entries with new users or product (no cold start)
    retData = data_orig.loc[data_orig['user_id'].isin(trainUsers)]
    retData = retData.loc[retData['business_id'].isin(trainProducts)]
    dataFilteredOut = data_orig.drop(retData.index)
    missingProducts = dataFilteredOut['business_id'].unique()
    missingUsers = dataFilteredOut['user_id'].unique()
    return retData, missingProducts, missingUsers


################################
# load train data
trainDataDF_all = load(trainData)
trainDataDF_all = cleanDataFromNulls(trainDataDF_all)
validationDataDF, trainDataDF, trainProducts, trainUsers = splitTrainValidation(trainDataDF_all)
################################
# load test data
testDataDF_orig = load(testData)
# Clean test from cold start
testDataDF, missingProducts, missingUsers = cleanColdStarts(trainProducts, trainUsers, testDataDF_orig)
################################
# run
# We add an option to run a few options - to have multiple parameters tests
# this was used to select the best lambda, Dalta and K
# (we also tried a few different ranges for the P, Q, Y random values matrix
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

                        ########   Predict SVD/SVD++ on Test   ########
                        predictBeginTime = time.time()
                        calculatedTestRates = mySvd.predictRates(testDataDF)
                        PredictTime = time.time() - predictBeginTime

                        if (ensamble):
                            ### sentiment prediciton
                            # build Sentiment Predictor using the same tarin data
                            x, y_train, y_test = prepareDataForSemantic(trainDataDF_all, testDataDF)
                            x = extract_features(x)
                            x_train, x_test = split_and_reduce(x)
                            x_train, x_test = apply_target_encoder(x_train, x_test, y_train, 'user_id', 1)
                            classifier = train_classifier(x_train, y_train)
                            ########   Predict using the semantic Predictor on Test   ########
                            sentimentModelRatesPrediction = classifier.predict(x_test)

                            ### Ensamble
                            ensamblePrediciton = (sentimentModelRatesPrediction + calculatedTestRates) / 2

                        ######## Calc Errors  ########
                        actuaTestRaes = testDataDF['stars'].to_list()
                        testRMSE = RMSE(actuaTestRaes, calculatedTestRates)
                        testMAE = MAE(actuaTestRaes, calculatedTestRates)

                        testSentimentRMSE = None
                        testSentimentMAE = None
                        testEnsambleRMSE = None
                        testEnsambleMAE = None
                        ensambleStringResults = ""

                        if (ensamble):
                            testSentimentRMSE = RMSE(actuaTestRaes, sentimentModelRatesPrediction)
                            testSentimentMAE = MAE(actuaTestRaes, sentimentModelRatesPrediction)
                            testEnsambleRMSE = RMSE(actuaTestRaes, ensamblePrediciton)
                            testEnsambleMAE = MAE(actuaTestRaes, ensamblePrediciton)
                            ensambleStringResults = "testSentimentRMSE[" + str(testSentimentRMSE) + "]" \
                                                    + "testSentimentMAE[" + str(testSentimentMAE) + "]" \
                                                    + "testEnsambleRMSE[" + str(testEnsambleRMSE) + "]" \
                                                    + "testEnsambleMAE[" + str(testEnsambleMAE) + "]"

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
                                          + ensambleStringResults
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
                                       + ensambleStringResults \
                                       + 'host[' + socket.gethostname() + "]"
                        printToFile(".\\results\\" + fileBaseName + ".log")
                        ##########################################
                        # Save the Model to be reused if needed
                        # dumpFileFullPath = ".\\dumps\\" + fileBaseName + ".dump"
                        # with open(dumpFileFullPath, 'wb') as fp:
                        #     pickle.dump(mySvd, fp)
                        ##########################################
                        # Load the Model to be reused - sample code
                        # mySvdload = None
                        # with open(dumpFileFullPath, 'rb') as fp:
                        #     mySvdload = pickle.load(fp)

printDebug(" ---- Done ---- ")
exit(0)
