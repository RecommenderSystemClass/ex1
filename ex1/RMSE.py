import numpy as np
import math

def RMSE(x,y):
    #print("RMSE x["+str(len(x))+"]y["+str(len(y))+"]")
    ret =  math.sqrt(np.square(np.subtract(x,y)).mean())
    #print("RMSE ret["+str(ret)+"]x["+str(len(x))+"]y["+str(len(y))+"]")
    return ret

#test RMSE
# rmse = RMSE([1,2,3,4],[1,2,3,4])
# rmse = RMSE([1,2,3,4],[0,3,2,5])
# rmse = RMSE([1,2,3,4],[-1,4,1,6])

