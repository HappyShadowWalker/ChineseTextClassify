__author__ = 'ShadowWalker'
from math import log
from numpy import *
# belowInfinity = float('-Inf')
belowInfinity = -10000000000000000
# 训练隐状态之间的转移概率
def trainTransProb(trainFileName, tranFileName):
    trainFile = open(trainFileName, 'r')
    trainFileContent = trainFile.read()
    trainFile.close()
    trainFileWords = trainFileContent.split()
    #print(len(trainFileWords))
    # 对训练文本中的词进行遍历，得到隐状态转移概率矩阵
    BTransform = [0, 0, 0, 0, 0]
    MTransform = [0, 0, 0, 0, 0]
    ETransform = [0, 0, 0, 0, 0]
    STransform = [0, 0, 0, 0, 0]
    #
    singleWordCount = 0
    complexWordCount = 0
    if len(trainFileWords[0]) == 1:
        STransform[4] = STransform[4] + 1
    elif len(trainFileWords[0]) >=2:
        BTransform[4] = BTransform[4] + 1
        ETransform[4] = ETransform[4] + 1
        MTransform[4] = MTransform[4] + len(trainFileWords[0]) -2
    for i in range(1, len(trainFileWords)):
        currentWord = trainFileWords[i]
        lastWord = trainFileWords[i-1]
        currentWordLen = len(currentWord)
        lastWordLen = len(lastWord)
        if currentWordLen == 1:
            singleWordCount = singleWordCount + 1
            STransform[4] = STransform[4] + 1
            if lastWordLen == 1:
                STransform[3] = STransform[3] + 1
            elif lastWordLen > 1:
                ETransform[3] = ETransform[3] + 1
        elif currentWordLen == 2:
            BTransform[4] = BTransform[4] + 1
            ETransform[4] = ETransform[4] + 1
            complexWordCount = complexWordCount + 1
            BTransform[2] = BTransform[2]+1
            if lastWordLen == 1:
                STransform[0] = STransform[0] + 1
            elif lastWordLen > 1:
                ETransform[0] = ETransform[0] +1
        elif currentWordLen > 2:
            BTransform[4] = BTransform[4] + 1
            ETransform[4] = ETransform[4] + 1
            MTransform[4] = MTransform[4] + currentWordLen - 2
            complexWordCount = complexWordCount + 1
            MTransform[1] = MTransform[1] + currentWordLen -3
            BTransform[1] = BTransform[1] + 1
            MTransform[2] = MTransform[2] + 1
            if lastWordLen == 1:
                STransform[0] = STransform[0] + 1
            elif lastWordLen > 1:
                ETransform[0] = ETransform[0] + 1

    # 计算初始概率分布
    BIniProb = log(complexWordCount/(singleWordCount+complexWordCount))
    print(BIniProb)
    SIniProb = log(singleWordCount/(singleWordCount+complexWordCount))
    print(SIniProb)
    for j in range(len(BTransform)-1):
        if BTransform[j] == 0:
            BTransform[j] = belowInfinity
        if MTransform[j] == 0:
            MTransform[j] = belowInfinity
        if ETransform[j] == 0:
            ETransform[j] = belowInfinity
        if STransform[j] == 0:
            STransform[j] = belowInfinity
        if BTransform[j] > 0:
            BTransform[j] = log(BTransform[j] / BTransform[4])
        if MTransform[j] > 0:
            MTransform[j] = log(MTransform[j] / MTransform[4])
        if ETransform[j] > 0:
            ETransform[j] = log(ETransform[j] / ETransform[4])
        if STransform[j] > 0:
            STransform[j] = log(STransform[j] / STransform[4])
    tranFile = open(tranFileName, 'w')
    tranFile.close()
    tranFile = open(tranFileName, 'r+')
    tranFile.write('#TransProbMatrix BMES\n')
    for k in range(4):
        tranFile.write(str(BTransform[k]))
        tranFile.write(" ")
    tranFile.write("\n")
    for k in range(4):
        tranFile.write(str(MTransform[k]))
        tranFile.write(" ")
    tranFile.write("\n")
    for k in range(4):
        tranFile.write(str(ETransform[k]))
        tranFile.write(" ")
    tranFile.write("\n")
    for k in range(4):
        tranFile.write(str(STransform[k]))
        tranFile.write(" ")
    tranFile.write("\n")
    tranFile.close()

# trainTransProb('G:\ChineseCut\PKU_GB\pku_training.txt', 'tran.txt')


def writeListToFile(dataList, fileName):
    listlen  = len(dataList)
    file = open(fileName, 'a')
    for i in range(listlen):
        file.write(str(dataList[i]))
        file.write(" ")
    file.write("\n")
    file.close()

# 通过训练数据得到发射概率矩阵
# 在这里采用加一平滑技术
def trainEmitProb(trainFileName, emitFileName):
    trainFile = open(trainFileName, 'r')
    trainFileContent = trainFile.read()
    trainFile.close()
    trainFileWords = trainFileContent.split()
    wordSet = set()
    for eachWord in trainFileWords:
        for eachSingleWord in eachWord:
            wordSet.add(eachSingleWord)

    # 把wordSet 中的单个字转成链表
    WordCount = len(wordSet)
    worddict = []
    for singleword in wordSet:
        worddict.append(singleword)
    BEmit = [1]*WordCount
    MEmit = [1]*WordCount
    EEmit = [1]*WordCount
    SEmit = [1]*WordCount
    BTotal = 0
    MTotal = 0
    ETotal = 0
    STotal = 0
    for eachWord in trainFileWords:
        wordlen = len(eachWord)
        if wordlen == 1:
            SEmit[worddict.index(eachWord)] = SEmit[worddict.index(eachWord)] + 1
            STotal = STotal + 1
        if wordlen == 2:
            BEmit[worddict.index(eachWord[0])] = BEmit[worddict.index(eachWord[0])] + 1
            EEmit[worddict.index(eachWord[1])] = EEmit[worddict.index(eachWord[1])] + 1
            BTotal = BTotal + 1
            ETotal = ETotal + 1
        if wordlen > 2:
            BTotal = BTotal + 1
            ETotal = ETotal + 1
            MTotal = MTotal + wordlen - 2
            BEmit[worddict.index(eachWord[0])] = BEmit[worddict.index(eachWord[0])] + 1
            EEmit[worddict.index(eachWord[wordlen-1])] = EEmit[worddict.index(eachWord[wordlen-1])] + 1
            for i in range(1, wordlen-1):
                MEmit[worddict.index(eachWord[i])] = MEmit[worddict.index(eachWord[i])] + 1

    for i in range(WordCount):
        BEmit[i] = log(BEmit[i]/(BTotal+WordCount))
        MEmit[i] = log(MEmit[i]/(MTotal+WordCount))
        EEmit[i] = log(EEmit[i]/(ETotal+WordCount))
        SEmit[i] = log(SEmit[i]/(STotal+WordCount))
    emitFile = open(emitFileName, 'w')
    emitFile.close()
    emitFile = open('worddict.txt', 'w')
    emitFile.write("#WordDict\n")
    # print(len(worddict))
    for eachWord in worddict:
        emitFile.write(eachWord)
        emitFile.write(" ")
    emitFile.write("\n")
    emitFile.close()
    writeListToFile("#B", emitFileName)
    writeListToFile(BEmit, emitFileName)
    writeListToFile("#M", emitFileName)
    writeListToFile(MEmit, emitFileName)
    writeListToFile("#E", emitFileName)
    writeListToFile(EEmit, emitFileName)
    writeListToFile("#S", emitFileName)
    writeListToFile(SEmit, emitFileName)
    # print(BEmit)
    # print(MEmit)
    # print(EEmit)
    # print(SEmit)
# trainEmitProb('G:\ChineseCut\PKU_GB\pku_training.txt', 'emit.txt')







