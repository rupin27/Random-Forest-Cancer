import numpy as np
import math
import matplotlib.pyplot as plt
import random
from collections import Counter
from utils import *

class TreeNode:
    def __init__(self, label, type):
        self.label = label
        self.type = type
        self.dataType = ""
        self.testAttr = ""
        self.edge = {}
        self.majority = -1
        self.thres = -1
        self.depth = 0
        self.parent = None

    def calcDepth(self):
        if self.parent is None:
            return 0
        else:
            return self.parent.calcDepth() + 1
    
def decisionTreeForest(dataSet: np.array, dictAttr: dict, impurity: str ='id3', maxDepth: int = 10, minSize: int = 10, minGain: float = 0.01):
    # Step 1: Prepare the data
    transDataset = np.copy(dataSet).T  
    sampDictAttr = dictAttr.copy()
    classIndex = list(dictAttr.values()).index("class")
    k = len(dictAttr) - 1
    randomList = random.sample(range(0, k), round(math.sqrt(k))) if classIndex != 0 else random.sample(range(1, k+1), round(math.sqrt(k)))
    randomList.append(classIndex)
    randomKey = [list(sampDictAttr.keys())[i] for i in randomList]
    trimDict = {key:sampDictAttr[key] for key in randomKey}
    trimData = np.array(transDataset[randomList])

    # Step 2: Check stopping criteria
    node = TreeNode(label = -1, type = "decision")
    currDepth = node.depth
    node.majority = majority(transDataset[classIndex])

    if same(transDataset[classIndex]):
        node.type = "leaf"
        node.label = transDataset[classIndex][0]
        return node
    
    if len(sampDictAttr) == 0:
        node.type = "leaf"
        node.label = majority(transDataset[classIndex])
        return node

    if len(dataSet) <= minSize:
        node.type = "leaf"
        node.label = majority(transDataset[classIndex])
        return node
    
    # Step 3: Choose the best attribute
    def processbest(algor):
        if algor == "cart":
            return bestSeparate(trimData.T, trimDict, 'gini')
        else: 
            return bestSeparate(trimData.T, trimDict, 'entropy')
        
    bestAttrDict, thresVal, gain = processbest(impurity)
    bestAttrName = list(bestAttrDict.keys())[0]
    bestAttrType = bestAttrDict[bestAttrName]
    node.dataType = bestAttrType
    node.testAttr = bestAttrName
    node.thres = thresVal
    bestAttrIndex = list(sampDictAttr.keys()).index(list(bestAttrDict.keys())[0])

    if gain < minGain:
        node.type = "leaf"
        node.label = majority(transDataset[classIndex])
        return node
    
    # Step 4: Split the data`
    splitDataLists = []
    if bestAttrType == "numerical":
        sortedDataset = transDataset.T[transDataset.T[:,bestAttrIndex].argsort(kind='quicksort')].T
        splitIndex = np.searchsorted(sortedDataset[bestAttrIndex], thresVal)
        splitDataListSamp = [sortedDataset.T[:splitIndex].T,sortedDataset.T[splitIndex:].T]
        for splitData in splitDataListSamp:
            splitData = np.delete(splitData, bestAttrIndex, 0)
            splitDataLists.append(splitData.T)
    else:
        highVal = list(Counter(transDataset[bestAttrIndex]).keys()) 
        for val in highVal:
            index = [idx for idx, element in enumerate(transDataset[bestAttrIndex]) if element == val]
            subdatav = np.array(transDataset.T[index]).T
            subdatav = np.delete(subdatav, bestAttrIndex, 0)  
            splitDataLists.append(subdatav.T) 
    sampDictAttr.pop(bestAttrName)

    # Step 5: Recursively build the tree
    edge = {}
    sdIndex = 0
    for subvdata in splitDataLists:
        if subvdata.size == 0:
            node.type = "leaf"
            node.label = node.majority
            node.thres = thresVal
            return node
        
        if node.calcDepth() + 1 > maxDepth:  
            node.type = "leaf"
            node.label = node.majority
            node.thres = thresVal
            return node 
        
        subtree = decisionTreeForest(subvdata, sampDictAttr, impurity, maxDepth, minSize, minGain)
        subtree.depth = currDepth + 1
        subtree.parent = node
            
        if bestAttrType == 'numerical':
            attributevalue = "<=" if sdIndex == 0 else ">"
        else:
            attributevalue = highVal[sdIndex]

        edge[attributevalue] = subtree
        sdIndex += 1
    node.edge = edge

    return node

# Predict the label of the test data, return correct and predict.
def predictLabel(tree: TreeNode, instance, sampDictAttr): # note that the instance is by row. (I formerly used by column)
    predict = tree.majority
    classIndex = list(sampDictAttr.values()).index("class")
    correct = instance[classIndex]
    if tree.type == 'leaf':
        predict = tree.label
        return predict, correct

    testindex = list(sampDictAttr.keys()).index(tree.testAttr)
    
    if tree.dataType == "numerical":
        nexttree = tree.edge['<='] if instance[testindex] <= tree.thres else tree.edge['>']
    else:
        if instance[testindex] not in tree.edge:
            return predict, correct
        nexttree = tree.edge[instance[testindex]]

    return predictLabel(nexttree, instance, sampDictAttr)