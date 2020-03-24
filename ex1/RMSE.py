import numpy as np
import math

def RMSE(x,y):
    ret =  math.sqrt(np.square(np.subtract(x,y)).mean())
    print("RMSE ["+str(ret)+"]x["+str(x)+"]y["+str(y)+"]")
    return ret

#test RMSE
# rmse = RMSE([1,2,3,4],[1,2,3,4])
# rmse = RMSE([1,2,3,4],[0,3,2,5])
# rmse = RMSE([1,2,3,4],[-1,4,1,6])

