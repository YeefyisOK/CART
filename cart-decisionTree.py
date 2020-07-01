#-*-coding:UTF-8-*-

from itertools import *
import json
import operator,time,math
import matplotlib.pyplot as plt
from numpy import *

# 计算gini系数
def calGini(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}#类别字典,类别序号类别数量
    for featVec in dataSet:
        currentLabel = featVec[-1]#数据集最后一列是类别
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    gini=1
    for label in labelCounts.keys():
        prop = float(labelCounts[label])/numEntries
        gini -= prop*prop
    return gini

def splitDataSet(dataSet, axis, value,threshold):#根据特征、特征值和方向划分数据集
    retDataSet = []
    if threshold == 'lt':
        for featVec in dataSet:
            if featVec[axis] <= value:
                retDataSet.append(featVec)
    else:
        for featVec in dataSet:
            if featVec[axis] > value:
                retDataSet.append(featVec)
    return retDataSet


# 返回最好的特征以及特征值
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    bestGiniGain = 1.0; bestFeature = -1; bsetValue = ""
    for i in range(numFeatures):        #遍历特征
        featList = [example[i] for example in dataSet]#得到特征列
        uniqueVals = list(set(featList))       #从特征列获取该特征的特征值的set集合
        uniqueVals.sort()
        for value in uniqueVals:# 遍历所有的特征值
            GiniGain = 0.0
            # 左子树基尼系数
            left_subDataSet = splitDataSet(dataSet, i, value,'lt')
            left_prob = len(left_subDataSet)/float(len(dataSet))
            GiniGain += left_prob * calGini(left_subDataSet)
            # 右子树基尼系数
            right_subDataSet = splitDataSet(dataSet, i, value,'gt')
            right_prob = len(right_subDataSet)/float(len(dataSet))
            GiniGain += right_prob * calGini(right_subDataSet)
            # print GiniGain
            if (GiniGain < bestGiniGain):      #当结果最优时，进行更新
                bestGiniGain = GiniGain      #更新最优结果及最优特征
                bestFeature = i
                bsetValue=value
    return bestFeature,bsetValue

# 多数表决
def majorityCnt(classList):
    classCount = {}#类别号 数据集里出现这个类别的数目
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 生成cart决策树
def createTree(dataSet,depth):#生成一棵指定深度的cart
    classList = [example[-1] for example in dataSet]
    if depth == 0:#如果到达指定深度，直接多数表决
        return majorityCnt(classList)
    if classList.count(classList[0]) == len(classList):
        return classList[0] # 所有数据 类别都一样，不需继续划分
    if len(dataSet) == 1: # 如果没有继续可以划分的特征，就多数表决决定分支的类别
        return majorityCnt(classList)
    bestFeat, bsetValue = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = str(bestFeat) + ":" + str(bsetValue) # 用最优特征+阈值作为节点  attribute[bestFeat]可以打印出节点
    if bestFeat == -1:
        return majorityCnt(classList)
    myTree = {bestFeatLabel : {}}
    myTree[bestFeatLabel]['<=' + str(round(float(bsetValue),3))] = createTree(splitDataSet(dataSet, bestFeat, bsetValue,'lt'),depth-1)
    myTree[bestFeatLabel]['>' + str(round(float(bsetValue),3))] = createTree(splitDataSet(dataSet, bestFeat, bsetValue,'gt'),depth-1)
    return myTree


def loadDataSet(fileName):
    # get number of fields
    numFeat = len(open(fileName).readline().split('\t'))
    numFeat=11
   # print ("numFeat")
   # print(numFeat)
    dataArr = []
    labelArr = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split(',')
        for i in range(numFeat):
            lineArr.append(int(curLine[i]))
            #print("lineArr")
            #print(lineArr)
        dataArr.append(lineArr)
        # labelArr.append(float(curLine[-1]))
    print("dataArr: ", dataArr)
    # print("labelArr: ", labelArr)
    return dataArr

def predict(tree,sample):
    if type(tree) is not dict:#一直到不是字典了，打印该类别
        return tree
    root = list(tree.keys())[0]
    feature,threshold=root.split(":")#取出一个节点，以冒号分成特征和值
    feature=int(feature)
    threshold=float(threshold)
    if sample[feature]>threshold:#递归预测
        return predict(tree[root]['>'+str(round(float(threshold),3))], sample)
    else:
        return predict(tree[root]['<='+str(round(float(threshold),3))], sample)

#用得到的决策树对测试集做预测
def cartClassify(dataMatrix,tree):
    errorList=ones((shape(dataMatrix)[0],1))# 返回预测对或者错(对为0，错为1，方便计算预测错误的个数)
    predictResult=[]#记录预测的结果
    classList = [example[-1] for example in dataMatrix]
    errnum=0
    for i in range(len(dataMatrix)):
        res=predict(tree,dataMatrix[i])
        errorList[i]=(res!=classList[i])
        if res!=classList[i]:
            errnum+=1
        predictResult.append([int(res)] )
        # print predict(tree,dataMatrix[i]),classList[i]
    return errnum,predictResult


if __name__ == "__main__":
    attribute=['牌1花色','牌1等级','牌2花色','牌2等级','牌3花色','牌3等级','牌4花色','牌4等级','牌5花色','牌5等级']
    dataArrtrain = loadDataSet("J:/data/poker-hand-training-true.data")
    #print('dataArr', dataArr)
    depth=15
    tree=createTree(dataArrtrain,depth)
    print("我的树")
    treech = json.dumps(tree, ensure_ascii=False)
    print(treech)
    datatest=loadDataSet("J:/data/datatest.txt")
    (errnum,predictResult)=cartClassify(datatest,tree)
    print("预测错误的个数")
    print(errnum)
    print("测试集个数")
    print(len(datatest))
    zhengquelv=1-float(errnum)/len(datatest)
    print("正确率")
    print(zhengquelv)
    print("预测结果")
    print (predictResult)
    # print(finallyPredictResult)


