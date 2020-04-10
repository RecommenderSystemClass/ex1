# Alex Danieli 317618718
# Gil Shamay 033076324

# Mean Absolute Error (MAE) is a popular alternative, given by
# MEA = (1/num of user-item pairs ) * sum of all absolute Errors
# Compared to MAE, RMSE disproportionately penalizes large errors, so that,
# given a test set with four hidden items
# RMSE would prefer a system that makes
# an error of 1 on three ratings and 0 on the fourth
# to one that makes an error of 3 on one rating and 0 on all three others,
# while MAE would prefer the second system.


import numpy as np
from RMSE import *

def MAE(x,y):
    #print("MSE x["+str(len(x))+"]y["+str(len(y))+"]")
    ret =  (np.abs(np.subtract(x,y))).mean()
    # print("MSE ret["+str(ret)+"]x["+str(len(x))+"]y["+str(len(y))+"]")
    return ret

#test MSE
# mae = MAE([1,2,3,4],[1,2,3,4])
# mae = MAE([1,2,3,4],[0,3,2,5])
# mae = MAE([1,2,3,4],[-1,4,1,6])
# mae = MAE([1,2,3,4],[5,2,3,0])
# mae = MAE([1,2,3,4],[4,2,3,4]) #1 of 3  RMSE
# mae = MAE([1,2,3,4],[2,3,4,4]) # 3 of 1  MEA

