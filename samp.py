#RANDOM FOReST.py

from utils import *
from decisionTree import *

# Random Forest, plant a forest of n trees
def plantforest(data, categorydict, ntree=10, maxdepth=10, minimalsize=10, minimalgain=0.01, algortype='id3', bootstrapratio = 0.1):
    def bootstrap(data, ratio=0.1): 
        n_samples = data.shape[0]
        sample_indices = np.random.choice(n_samples, size=int(n_samples * (1 - ratio)), replace=False)
        return data[sample_indices]
    forest = []
    for i in range(ntree):
        datause = bootstrap(data, bootstrapratio)
        tree = decisiontreeforest(datause, categorydict, algortype, maxdepth, minimalsize, minimalgain)
        forest.append(tree)
    return forest

# Predict the class of a single instance
def forestvote(forest, instance, categorydict):
    votes = {}
    for tree in forest:
        predict, correct= prediction(tree,instance,categorydict)
        if predict not in votes:
            votes[predict] = 1
        else:
            votes[predict] += 1
    return max(votes, key=votes.get), correct

# A complete k-fold cross validation
def kfoldcrossvalid(data, categorydict, k=10, ntree=10, maxdepth=5, minimalsize=10, minimalgain=0.01, algortype='id3', bootstrapratio=0.1):

    def stratifiedkfold(data, categorydict, k=10):
        classindex = list(categorydict.values()).index("class")
        datacopy = np.copy(data)
        np.random.shuffle(datacopy)

        classes, counts = np.unique(datacopy[:, classindex], return_counts=True)
        nclass = len(classes)
        counts = counts / np.sum(counts)

        folds = np.zeros(len(datacopy), dtype=np.int32)

        for i in range(nclass):
            class_indices = np.where(datacopy[:, classindex] == classes[i])[0]
            for j, fold in enumerate(np.array_split(class_indices, k)):
                folds[fold] = j

        return [datacopy[folds == i] for i in range(k)]
    
    listofnd = []
    accuracylist = []
    folds = stratifiedkfold(data, categorydict, k)
    for i in range(k):
        testdataset = folds[i]
        traindataset = np.concatenate([folds[j] for j in range(k) if j != i])
        correctcount = 0
        trainforest = plantforest(traindataset, categorydict, ntree, maxdepth, minimalsize, minimalgain, algortype, bootstrapratio)
        emptyanalysis = []
        for instance in testdataset:
            predict, correct = forestvote(trainforest,instance,categorydict)
            emptyanalysis.append([predict, correct])
            if predict == correct:
                correctcount += 1
        listofnd.append(np.array(emptyanalysis))

        accuracylist.append(correctcount/len(testdataset))
    acc = np.mean(accuracylist)
    return listofnd, acc




# DECISION TREE.py
from utils import *
import numpy as np
import math
import random
from collections import Counter

class Treenode:

    def __init__(self, label, nodeType):
        self.label = label
        self.nodeType = nodeType
        self.majority = -1
        self.datatype = ""
        self.testattribute = ""
        self.threshold = -1
        self.edge = {}
        self.depth = 0
        self._caldepth = 0
        self.parent = None

    def caldepth(self):
        if self.parent is None:
            return 0
        else:
            return self.parent.caldepth() + 1
    
# Decision Tree that only analyze square root of the data.
def decisiontreeforest(dataset: np.array, dictattributes: dict, algortype: str ='id3', maxdepth: int = 10, minimalsize: int = 10, minimalgain: float = 0.01):
    # Step 1: Prepare the data
    datasetcopy = np.copy(dataset).T # dataset copy is by colomn. 
    dictattricopy = dictattributes.copy()
    classindex = list(dictattributes.values()).index("class")
    k = len(dictattributes)-1
    randomlist = random.sample(range(0, k), round(math.sqrt(k))) if classindex !=0 else random.sample(range(1, k+1), round(math.sqrt(k)))
    randomlist.append(classindex)
    randomkey = [list(dictattricopy.keys())[i] for i in randomlist]
    trimmeddict = {key:dictattricopy[key] for key in randomkey}
    trimmeddata = np.array(datasetcopy[randomlist])

    # Step 2: Check stopping criteria
    node = Treenode(label=-1,nodeType="decision")
    currentdepth = node.depth
    node.majority = majority(datasetcopy[classindex])

    if same(datasetcopy[classindex]):
        node.nodeType = "leaf"
        node.label = datasetcopy[classindex][0]
        return node

    if len(dictattricopy) == 0 or len(dataset) <= minimalsize or node.caldepth()+1 > maxdepth:
        node.nodeType = "leaf"
        node.label = majority(datasetcopy[classindex])
        return node

    # Step 3: Choose the best attribute
    def processbest(algor):
        if algor == "gini":
            return bestseperate(trimmeddata.T, trimmeddict, "id3")
        else:
            return bestseperate(trimmeddata.T, trimmeddict, "id3")

    bestattributedict, thresholdval, gain = processbest(algortype)
    bestattributename = list(bestattributedict.keys())[0]
    bestattributetype = bestattributedict[bestattributename]
    node.datatype = bestattributetype
    node.testattribute = bestattributename
    node.threshold = thresholdval
    bindex = list(dictattricopy.keys()).index(list(bestattributedict.keys())[0])

    if gain < minimalgain:
        node.nodeType = "leaf"
        node.label = majority(datasetcopy[classindex])
        return node

    # Step 4: Split the data
    subdatalists = []
    if bestattributetype == "numerical":
        sortedcopy = datasetcopy.T[datasetcopy.T[:,bindex].argsort(kind='quicksort')].T
        splitindex = np.searchsorted(sortedcopy[bindex], thresholdval)
        subdatalistraw = [sortedcopy.T[:splitindex].T,sortedcopy.T[splitindex:].T]
        for subdata in subdatalistraw:
            subdata = np.delete(subdata,bindex,0)
            subdatalists.append(subdata.T)
    else:
        bigv = list(Counter(datasetcopy[bindex]).keys())
        for smallv in bigv:
            index = [idx for idx, element in enumerate(datasetcopy[bindex]) if element == smallv]
            subdatav = np.array(datasetcopy.T[index]).T
            subdatav = np.delete(subdatav,bindex,0)
            subdatalists.append(subdatav.T)
    dictattricopy.pop(bestattributename)

    # Step 5: Recursively build the tree
    edge = {}
    sdindex = 0
    for subvdata in subdatalists:
    
        subtree = decisiontreeforest(subvdata, dictattricopy, algortype, maxdepth, minimalsize, minimalgain)
        subtree.depth = currentdepth + 1
        subtree.parent = node
    
        if bestattributetype == 'numerical':
            attributevalue = "<=" if sdindex == 0 else ">"
        else:
            attributevalue = bigv[sdindex]
        edge[attributevalue] = subtree
        sdindex += 1

    node.edge = edge

    return node

# Predict the label of the test data, return correct and predict.
def prediction(tree: Treenode, instance, dictattricopy): # note that the instance is by row. (I formerly used by column)
 
    predict = tree.majority
    classindex = list(dictattricopy.values()).index("class")
    correct = instance[classindex]

    if tree.nodeType == 'leaf':
        predict = tree.label
        return predict, correct

    testindex = list(dictattricopy.keys()).index(tree.testattribute)

    if tree.datatype == "numerical":
        if instance[testindex] <= tree.threshold:
            nexttree = tree.edge['<=']
        else:
            nexttree = tree.edge['>']
    else:
        if instance[testindex] not in tree.edge:
            return predict, correct

        nexttree = tree.edge[instance[testindex]]
    return prediction(nexttree, instance, dictattricopy)


#UTILS.py

import numpy as np
import math

def same(attributecolumn):
    return all(item == attributecolumn[0] for item in attributecolumn)

def majority(attributecolumn):
    return np.argmax(np.bincount(attributecolumn.astype(int)))

def bestseperate(dataset, attributes:dict, type:str):
    def id3EntropyCalc(datColumn):
    # initialize a dictionary to count the occurrences of each value in the input column
        cnts = {}
        for val in datColumn:
            cnts[val] = cnts.get(val, 0) + 1
        # calculate entropy
        fEntropy = sum(-count / len(datColumn) * math.log2(count / len(datColumn)) for count in cnts.values())
        return fEntropy
    
    def giniIndexCalc(datColumn):
        # initialize a dictionary to count the occurrences of each value in the input column
        cnts = {}
        for val in datColumn:
            cnts[val] = cnts.get(val, 0) + 1
        tCnt = len(datColumn)
        # compute the gini impurity value
        giniIndexVal = 1 - sum((count / tCnt) ** 2 for count in cnts.values())
        return giniIndexVal

    dataset = np.array(dataset)
    classIndex = list(attributes.values()).index("class")
    if (type == "id3"):
        origEntropy = id3EntropyCalc(dataset[:, classIndex])
    else:
        origEntropy = giniIndexCalc(dataset[:, classIndex])
    bestAttr = {}
    thresVal = -1
    smallestEntropy = origEntropy

    for i, attribute in enumerate(attributes):
        if attribute == "class":
            continue
        idx = i if i < classIndex else i + 1
        attrType = attributes[attribute]

        if attrType in ["categorical", "binary"]:
            categList = np.unique(dataset[:, idx])
            currEntropy = 0

            for category in categList:
                indices = np.where(dataset[:, idx] == category)[0]
                subset = dataset[indices, classIndex]
                mult = id3EntropyCalc(subset) if type == "id3" else giniIndexCalc(subset)
                currEntropy += len(subset) / len(dataset) * mult

            if currEntropy < smallestEntropy:
                smallestEntropy = currEntropy
                bestAttr = {attribute: attrType}

        elif attrType == "numerical":
            thresholds = np.unique(dataset[:, idx])
            if len(thresholds) == 1:
                continue

            thresholds = (thresholds[1:] + thresholds[:-1]) / 2
            for threshold in thresholds:
                leftIndices = np.where(dataset[:, idx] <= threshold)[0]
                rightIndices = np.where(dataset[:, idx] > threshold)[0]
                leftSubset = dataset[leftIndices, classIndex]
                rightSubset = dataset[rightIndices, classIndex]
                if (type == "id3"):
                    leftSubEnt = id3EntropyCalc(leftSubset)
                    rightSubEnt = id3EntropyCalc(rightSubset)
                else:
                    leftSubEnt = giniIndexCalc(leftSubset)
                    rightSubEnt = giniIndexCalc(rightSubset)                
                currEntropy = (len(leftSubset) / len(dataset)) * leftSubEnt + \
                               (len(rightSubset) / len(dataset)) * rightSubEnt

                if currEntropy < smallestEntropy:
                    smallestEntropy = currEntropy
                    thresVal = threshold
                    bestAttr = {attribute: attrType}

    gain = origEntropy - smallestEntropy
    return bestAttr, thresVal, gain



#RUN.py

from utils import *
from randomForest import *
import csv
    
def importdata(filename: str, delimiter: str, class_column: str, categorydict: dict):
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False
    
    file = open("datasets/" + filename, encoding="utf-8-sig")
    reader = csv.reader(file, delimiter=delimiter)
    dataset = []
    for row in reader:
        dataset.append(row)
    file.close()

    categorydict[class_column] = "class"
    for i, category in enumerate(dataset[0]):
        if category not in categorydict:
            categorydict[category] = "numerical" if is_number(dataset[1][i]) else "categorical"

    data = np.array(dataset[1:]).astype(float)
    return data, categorydict

def importcmcdata():
    file = open("datasets/cmc.data", encoding='utf-8-sig')
    reader = csv.reader(file, delimiter=',')
    dataset = []
    for row in reader:
        dataset.append(row)
    file.close()
    cmccategory = {"Wife's age":"numerical","Wife's education":"categorical",
                   "Husband's education":"categorical","Number of children ever born":"numerical",
                   "Wife's religion":"binary","Wife's now working?":"binary",
                   "Husband's occupation":"categorical","Standard-of-living index":"categorical",
                   "Media exposure":"binary","Contraceptive method used":"class"}
    
    cmcdata = np.array(dataset).astype(int)
    return cmcdata, cmccategory

if __name__=="__main__":
    cmcdata,cmccategory = importcmcdata()
    lists, acc = kfoldcrossvalid(cmcdata,cmccategory, k=10, ntree=20, maxdepth=10, minimalsize=10, minimalgain=0.01, algortype='id3', bootstrapratio = 0.1)
    print(acc)
    print(lists)

#EVALUATION MATRIX.py

# Evaluation Metrics

def accuracy(truePosi, trueNega, falsePosi, falseNega):
	return (truePosi + trueNega) / (truePosi + trueNega + falseNega + falsePosi)

def precision(truePosi, falsePosi):
	if (truePosi + falsePosi) == 0:
		return 0
	return truePosi / (truePosi + falsePosi)

def recall(truePosi, falseNega):
	if (truePosi + falseNega) == 0:
		return 0
	return truePosi / (truePosi + falseNega)

def fscore(truePosi, falsePosi, falseNega, beta = 1):
    if not (truePosi or falsePosi or falseNega):
        return 0
    precision = truePosi / (truePosi + falsePosi)
    recall = truePosi / (truePosi + falseNega)
    fScore = (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall)
    return fScore


def evaluate(listsofoutput, positivelabel, beta=1):
    accList, preList, recList, f1List = [], [], [], []
    for output in listsofoutput:
        tp = sum(p == a == positivelabel for p, a in output)
        tn = sum(p == a != positivelabel for p, a in output)
        fp = sum(p == positivelabel and a != positivelabel for p, a in output)
        fn = sum(p != positivelabel and a == positivelabel for p, a in output)
        accList.append(accuracy(tp, tn, fp, fn))
        preList.append(precision(tp, fp))
        recList.append(recall(tp, fn))
        f1List.append(fscore(tp, fp, fn, beta))
    return accList, preList, recList, f1List

def meanevaluation(listsofoutput, positivelabel, beta=1):
    accuarcylists, precisionlists, recalllists, fscorelists = evaluate(listsofoutput, positivelabel, beta)
    return sum(accuarcylists)/len(accuarcylists), sum(precisionlists)/len(precisionlists), sum(recalllists)/len(recalllists), sum(fscorelists)/len(fscorelists)

def printMetrics(acc, pre, rec, fsc, beta, nvalue, title):
    acc, pre, rec, fsc = round(acc, 3), round(pre, 3), round(rec, 3), round(fsc, 3)
    print(f"Result/Stat of {nvalue} trees random forest of {title}:")
    print(f"Accuracy: {acc}")
    print(f"Precision: {pre}")
    print(f"Recall: {rec}")
    print(f"F-Score, Beta={beta}: {fsc}")