# Alex Danieli 317618718
# Gil Shamay 033076324
#Root Mean Squared Error (RMSE) is perhaps the most popular metric used
# in evaluating accuracy of predicted ratings. The system generates predicted ratings
# rui for a test set T of user-item pairs (u,i) for which the true ratings rui are known.
# Typically, rui are known because they are hidden in an offline experiment, or because
# they were obtained through a user study or online experiment. The RMSE between
# the predicted and actual ratings is given by:
# RMSE = the Root of [(1/num of user-item pairs ) * (sum of all ((absolute Errors)^2)]

import numpy as np
import math

def RMSE(x,y):
    # print("RMSE x["+str(len(x))+"]y["+str(len(y))+"]")
    ret =  math.sqrt(np.square(np.subtract(x,y)).mean())
    # print("RMSE ret["+str(ret)+"]x["+str(len(x))+"]y["+str(len(y))+"]")
    return ret

#test RMSE
# rmse = RMSE([1,2,3,4],[1,2,3,4])
# rmse = RMSE([1,2,3,4],[0,3,2,5])
# rmse = RMSE([1,2,3,4],[-1,4,1,6])
# rmse = RMSE([1,2,3,4],[5,2,3,0])
# rmse = RMSE([1,2,3,4],[4,2,3,4])
# rmse = RMSE([1,2,3,4],[2,3,4,4])

