__author__ = 'ShadowWalker'
import codecs
import math
import sys
# 使用开方检验选择特征
# 按UTF-8编码格式读取文件

# 定义停止词
def ignore(s):
    return s == 'nbsp' or s == ' ' or s == ' ' or s == '/t' or s == '/n' \
           or s == '，' or s == '。' or s == '！' or s == '、' or s == '―'\
           or s == '？'  or s == '＠' or s == '：' \
           or s == '＃' or s == '%'  or s == '＆' \
           or s == '（' or s == '）' or s == '《' or s == '》' \
           or s == '［' or s == '］' or s == '｛' or s == '｝' \
           or s == '*' or s == ',' or s == '.'  or s == '&' \
           or s == '!' or s == '?' or s == ':' or s == ';'\
           or s == '-' or s == '&'\
           or s == '<' or s == '>' or s == '(' or s == ')' \
           or s == '[' or s == ']' or s == '{' or s == '}' or s == 'nbsp10' or s == '3.6' or s=='about' or s =='there' \
           or s == "see" or s == "can" or s == "U" or s == "L" or s == " " or s == "in" or s ==";" or s =="a" or s =="0144"\
           or s == "\n" or s == "our"

# print(stopwords)

# 对卡方检验所需的 a b c d 进行计算
# a：在这个分类下包含这个词的文档数量
# b：不在该分类下包含这个词的文档数量
# c：在这个分类下不包含这个词的文档数量
# d：不在该分类下，且不包含这个词的文档数量

#
ClassCode = ['C000007', 'C000008', 'C000010', 'C000013', 'C000014', 'C000016', 'C000020', 'C000022', 'C000023', 'C000024']

# 构建每个类别的词Set

# 分词后的文件路径
# textCutBasePath = "G:\\ChineseTextClassify\\SogouCCut\\"
textCutBasePath = sys.path[0] + "\\SogouCCut\\"
# 构建每个类别的词向量
def buildItemSets(classDocCount):
    termDic = dict()
    # 每个类别下的文档集合用list<set>表示, 每个set表示一个文档，整体用一个dict表示
    termClassDic = dict()
    for eachclass in ClassCode:
        currClassPath = textCutBasePath+eachclass+"\\"
        eachClassWordSets = set()
        eachClassWordList = list()
        for i in range(classDocCount):
            eachDocPath = currClassPath+str(i)+".cut"
            eachFileObj = open(eachDocPath, 'r')
            eachFileContent = eachFileObj.read()
            eachFileWords = eachFileContent.split(" ")
            eachFileSet = set()
            for eachword in eachFileWords:
                # 判断是否是停止词
                stripeachword = eachword.strip(" ")
                if not ignore(eachword) and len(stripeachword) > 0:
                    eachFileSet.add(eachword)
                    eachClassWordSets.add(eachword)
            eachClassWordList.append(eachFileSet)
            # print(eachFileSet)
        termDic[eachclass] = eachClassWordSets
        termClassDic[eachclass] = eachClassWordList
    return termDic, termClassDic



# 对得到的两个词典进行计算，可以得到a b c d 值
# K 为每个类别选取的特征个数

# 卡方计算公式
def ChiCalc(a, b, c, d):
    result = float(pow((a*d - b*c), 2)) /float((a+c) * (a+b) * (b+d) * (c+d))
    return result

def featureSelection(termDic, termClassDic, K):
    termCountDic = dict()
    for key in termDic:
        classWordSets = termDic[key]
        classTermCountDic = dict()
        for eachword in classWordSets:  # 对某个类别下的每一个单词的 a b c d 进行计算
            a = 0
            b = 0
            c = 0
            d = 0
            for eachclass in termClassDic:
                if eachclass == key: #在这个类别下进行处理
                    for eachdocset in termClassDic[eachclass]:
                        if eachword in eachdocset:
                            a = a + 1
                        else:
                            c = c + 1
                else: # 不在这个类别下进行处理
                    for eachdocset in termClassDic[eachclass]:
                        if eachword in eachdocset:
                            b = b + 1
                        else:
                            d = d + 1
            # print("a+c:"+str(a+c)+"b+d"+str(b+d))
            eachwordcount = ChiCalc(a, b, c, d)
            # print(eachwordcount)
            classTermCountDic[eachword] = eachwordcount
        # 对生成的计数进行排序选择前K个
        # 这个排序后返回的是元组的列表
        sortedClassTermCountDic = sorted(classTermCountDic.items(), key=lambda d:d[1], reverse=True)
        count = 0
        subDic = dict()
        for i in range(K):
            subDic[sortedClassTermCountDic[i][0]] = sortedClassTermCountDic[i][1]
        termCountDic[key] = subDic
    return termCountDic
        # print(sortedClassTermCountDic)


def writeFeatureToFile(termCountDic , fileName):
    featureSet = set()
    for key in termCountDic:
        for eachkey in termCountDic[key]:
            featureSet.add(eachkey)
    count = 1
    file = open(fileName, 'w')
    for feature in featureSet:
        # 判断feature 不为空
        stripfeature = feature.strip(" ")
        if len(stripfeature) > 0 and feature != " " :
            file.write(str(count)+" " +feature+"\n")
            count = count + 1
            print(feature)
    file.close()

# 调用buildItemSets
# buildItemSets形参表示每个类别的文档数目,在这里训练模型时每个类别取前200个文件
termDic, termClassDic = buildItemSets(200)
termCountDic = featureSelection(termDic, termClassDic, 1000)
writeFeatureToFile(termCountDic, "SVMFeature.txt")















