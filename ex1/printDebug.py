# Alex Danieli 317618718
# Gil Shamay 033076324


stringToPrintToFile = ""
def printDebug(str):
    global stringToPrintToFile
    print(str)
    stringToPrintToFile = stringToPrintToFile + str + "\r"

def printToFile(fileName):
    global stringToPrintToFile
    file1 = open(fileName,"a")
    file1.write("\r************************\r")
    file1.write(stringToPrintToFile)
    stringToPrintToFile =""
    file1.close()
