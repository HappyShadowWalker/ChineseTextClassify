__author__ = 'ShadowWalker'
from ChineseSegmentation.Viterbi import *
import os

# 测试分词算法，基于HMM模型的分词算法
# ChineseCutStr("我是一个中国人")
# 对搜狗预料库中的文本进行分词处理

folderList = ['C000007', 'C000008', 'C000010', 'C000013', 'C000014', 'C000016', 'C000020', 'C000022','C000023', 'C000024']
folderTextCount = 300 # 在这里只选择搜狗预料库中的每个分类下的300个文本进行分词处理
# readFilePathPrefix = "G:\\ChineseTextClassify\\SogouC\\ClassFile\\"
readFilePathPrefix = os.path.dirname(os.path.abspath('__file__')).strip("ChineseSegmentation")+"SogouC\\ClassFile\\"
# writeFilePathPrefix = "G:\\ChineseTextClassify\\SogouCCut\\"
writeFilePathPrefix = os.path.dirname(os.path.abspath('__file__')).strip("ChineseSegmentation") + "SogouCCut\\"
def cutText():
    for eachFolder in folderList:
        for i in range(0, folderTextCount):
            readfilename = readFilePathPrefix+eachFolder+"\\"+str(i)+".txt"
            writefilename = writeFilePathPrefix+eachFolder+"\\"+str(i)+".cut"
            ChineseCut(readfilename, writefilename)

cutText()
print("文本分词预处理成功！")



