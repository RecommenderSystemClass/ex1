import csv
import time
import pandas as pd

path = './data/trainData.csv'
data = None
chunksNum = 0

def processDataChunk(dataChunk):
    global chunksNum
    global data
    # we need for teh main process onl;y user_id,business_id,stars - we can drop all the rest to save mem
    dataChunk.drop(['text'], axis=1, inplace=True)
    dataChunk.drop(dataChunk.columns[[0]], axis=1, inplace=True)
    if (data is None):
        data = dataChunk
    else:
        data = data.append(dataChunk, ignore_index=True)

    if (chunksNum % 10 == 0):
        took = time.time() - beginTime
        print(str(chunksNum) + " " + str(took))
    chunksNum += 1
    return


beginTime = time.time()
pd.read_csv(path, chunksize=20000)
for dataChunk in pd.read_csv(path, chunksize=20000):
    processDataChunk(dataChunk)

took = time.time() - beginTime
print("chunksNum]" + str(chunksNum) + "]took[" + str(took) + "]data[" + str(data.count) + "]")
