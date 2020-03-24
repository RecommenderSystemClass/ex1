import csv
import time
import pandas as pd

def processDataChunk(dataChunk):
    # we need for the main process onl;y user_id,business_id,stars - we can drop all the rest to save mem
    dataChunk.drop(['text'], axis=1, inplace=True)
    dataChunk.drop(dataChunk.columns[[0]], axis=1, inplace=True)
    return dataChunk

def load(path):
    chunksNum = 0
    beginTime = time.time()
    data = None
    pd.read_csv(path, chunksize=20000)
    for dataChunk in pd.read_csv(path, chunksize=20000):
        dataChunk = processDataChunk(dataChunk)
        if (data is None):
            data = dataChunk
        else:
            data = data.append(dataChunk, ignore_index=True)

        if (chunksNum % 10 == 0):
            took = time.time() - beginTime
            print(str(chunksNum) + " " + str(took))
        chunksNum += 1
    took = time.time() - beginTime
    print("chunksNum]" + str(chunksNum) + "]took[" + str(took) + "]data[" + str(data.count) + "]")
    return data
