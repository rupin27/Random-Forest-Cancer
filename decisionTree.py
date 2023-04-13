import numpy as np
import math
import random
from collections import Counter

# decision tree algorithm to find the best way to separate a dataset
def bestSeparate(dataSet, attrs:dict, impurity:str):
    # helper function to find entropy
    def entropy(attrCol):
        # count of each unique value in attr column
        vals = Counter(attrCol).vals()
        entropy = 0
        for value in vals:
            # calc probabilty 
            j = (value / len(attrCol))
            # calc entropy
            entropy += -j * math.log2(j)
        return entropy
    # helper function to find gini
    def gini(attrCol):
        # get count of each unique value in attr column
        vals = Counter(attrCol).vals()
        giniVal = 1 - sum((value / len(attrCol))**2 for value in vals)
        return giniVal
    # transposing the dataset to get attr cols
    datasetCol = dataSet.T
    # index of class attr
    classIndex = list(attrs.vals()).index("class")
    
    # set original impurity measure
    if impurity == 'entropy':
        origImp = entropy(datasetCol[classIndex])
        minImp = origImp
    elif impurity == 'gini':
        origImp = gini(datasetCol[classIndex])
        minImp = origImp
    else:
        raise ValueError("invalid measure")
    # initialize threshold value and index counter
    thresVal = -1
    i = 0
    # setting intially the first attr as the best attr
    bestAttr = {list(attrs.keys())[i]:attrs[list(attrs.keys())[i]]}
    # check if class attr is not the first attr
    currAttr = list(attrs.keys())[1:] if (classIndex == 0) else list(attrs.keys())[:classIndex]
    # looping through all the attr excpet class
    for attr in currAttr:
        # setting column index based on class attr
        index = i + 1 if classIndex == 0 else i
        # separates the dataset into two categories based on each possible value.
        if attrs[attr] == "binary" or attrs[attr] == "categorical" :
            # gets all unique keys in the column of the current attribute.
            listKeys = list(Counter(datasetCol[index]).keys())
            # categ based on each unique key
            listCateg = [] 
            # go through ach unique key in the column of the current attribute
            for key in listKeys:
                listIndex = [idex for idex, element in enumerate(datasetCol[index]) if element == key]
                # get the category based on key
                category = np.array(datasetCol[classIndex][listIndex])
                listCateg.append(category)

            currImpurity = 0
            # calculates the impurity of each category and adds it to the current impurity.
            for category in listCateg:
                a = len(category) / len(datasetCol[index])
                if impurity == 'entropy':
                    currImpurity += a * entropy(category)
                elif impurity == 'gini':
                    currImpurity += a * gini(category)
            # current impurity is smaller than the smallest impurity so far
            if currImpurity < minImp:
                # update the smallest attr
                minImp = currImpurity
                # set the best attr to curr attr
                bestAttr = {attr: attrs[attr]}
        # in this case we find the best threshold value to separate the dataset
        elif attrs[attr] == "numerical":
            # sort the dataset by the column of the current attribute
            sortedDataset = datasetCol.T[datasetCol.T[:, index].argsort(kind='quicksort')].T
            # threshold value to the midpoint between the two smallest values
            currThres = (sortedDataset[index][1] + sortedDataset[index][0]) / 2
            j = 1
            while j < len(sortedDataset.T):
                # middle val between current value and the previous value
                currThres = (sortedDataset[index][j] + sortedDataset[index][j - 1]) / 2
                # seperate into two 
                listCateg = [sortedDataset[classIndex][:j], sortedDataset[classIndex][j:]]
                currImpurity = 0
                # calculates the impurity of each category and adds it to the current impurity.
                for category in listCateg:
                    a = len(category) / len(datasetCol[index])
                    if impurity == 'entropy':
                        currImpurity += a * entropy(category)
                    elif impurity == 'gini':
                        currImpurity += a * gini(category)
                # check if current impurity is smaller than the smallest impurity so far
                if currImpurity < minImp:
                    # update smallest impurity and thres
                    minImp = currImpurity
                    thresVal = currThres
                    bestAttr = {attr: attrs[attr]}
                j += 1
        i += 1
    netGain = origImp - minImp
    return bestAttr, thresVal, netGain
#________________________________________________________________________________________________________________________________________________________________________

class TreeNode:
    def __init__(self, label, type):
        self.label = label
        self.type = type
        self.dataType = ""
        self.sampAttr = ""
        self.edge = {}
        self.parent = None
        self.depth = 0
        self.thres = -1
        self.majority = -1
#________________________________________________________________________________________________________________________________________________________________________
# calculate the depth from a node
    def calcDepth(self):
        if self.parent is None:
            return 0
        else:
            # recursive
            return self.parent.calcDepth() + 1
    
def decisionTreeForest(dataSet, dictAttr, impurity: str ='id3', maxDepth: int = 10, minSize: int = 10, minGain: float = 0.01):
    # Step 1: Prepare the data
    # transposing and making a copy of original data and dictionary
    transDataset = np.copy(dataSet).T  
    sampDictAttr = dictAttr.copy()
    # index of the class label column
    classIndex = list(dictAttr.values()).index("class")
    # number of features
    k = len(dictAttr) - 1
    # get a random subset of features including the class label
    randomList = random.sample(range(0, k), round(math.sqrt(k))) if classIndex != 0 else random.sample(range(1, k+1), round(math.sqrt(k)))
    randomList.append(classIndex)
    # names of the selected features
    randomKey = [list(sampDictAttr.keys())[i] for i in randomList]
    # new dataset and dictionary containing only the selected features
    trimDict = {key:sampDictAttr[key] for key in randomKey}
    trimData = np.array(transDataset[randomList])

    # Step 2: Check stopping criteria

    def majClassDataset(attrCol):
        return np.argmax(np.bincount(attrCol.astype(int)))
    
    # default node
    node = TreeNode(label = -1, type = "decision")
    # get curr depth
    currDepth = node.depth
    # find the majority class in the dataset
    node.majority = majClassDataset(transDataset[classIndex])

    # check if all the samples have the same class label
    def checkSameLabel(attrCol):
        return all(attr == attrCol[0] for attr in attrCol)

    if checkSameLabel(transDataset[classIndex]):
        node.type = "leaf"
        node.label = transDataset[classIndex][0]
        return node
    # check if there are no more attributes to split on
    if len(sampDictAttr) == 0:
        node.type = "leaf"
        node.label = majClassDataset(transDataset[classIndex])
        return node
    # check if the number of samples is less than or equal to the minimum size
    if len(dataSet) <= minSize:
        node.type = "leaf"
        node.label = majClassDataset(transDataset[classIndex])
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
    node.sampAttr = bestAttrName
    node.thres = thresVal
    bestAttrIndex = list(sampDictAttr.keys()).index(list(bestAttrDict.keys())[0])

    if gain < minGain:
        node.type = "leaf"
        node.label = majClassDataset(transDataset[classIndex])
        return node
    
    # Step 4: Split the data`
    splitDataLists = []
    # chech if attr is numerical 
    if bestAttrType == "numerical":
        # sort the database based on the attr
        sortedDataset = transDataset.T[transDataset.T[:,bestAttrIndex].argsort(kind='quicksort')].T
        # split the datat using threshold val
        splitIndex = np.searchsorted(sortedDataset[bestAttrIndex], thresVal)
        splitDataListSamp = [sortedDataset.T[:splitIndex].T,sortedDataset.T[splitIndex:].T]
        for splitData in splitDataListSamp:
            splitData = np.delete(splitData, bestAttrIndex, 0)
            splitDataLists.append(splitData.T)
    else:
        # if the best attribute is categorical
        highVal = list(Counter(transDataset[bestAttrIndex]).keys()) 
        for val in highVal:
            # split the dataset into subsets for each category of the attr
            index = [idx for idx, element in enumerate(transDataset[bestAttrIndex]) if element == val]
            subdatav = np.array(transDataset.T[index]).T
            subdatav = np.delete(subdatav, bestAttrIndex, 0)  
            splitDataLists.append(subdatav.T) 
    # removing the best attribute from the sample dictionary
    sampDictAttr.pop(bestAttrName)

    # Step 5: Recursively build the tree
    edge = {}
    sdIndex = 0
    for subvdata in splitDataLists:
        # if the subset is empty ie. the tree has reached maximum depth, create a leaf node and return it
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
        # recursively build the subtree using the subset
        subtree = decisionTreeForest(subvdata, sampDictAttr, impurity, maxDepth, minSize, minGain)
        subtree.depth = currDepth + 1
        subtree.parent = node
        # check for numerical value
        if bestAttrType == 'numerical':
            # create an edge between the node and the subtree using the best attribute and its value
            attributevalue = "<=" if sdIndex == 0 else ">"
        else:
            attributevalue = highVal[sdIndex]

        edge[attributevalue] = subtree
        sdIndex += 1
    node.edge = edge

    return node
#________________________________________________________________________________________________________________________________________________________________________

# Predict the label of the test data, return correct and predict.
def predictLabel(tree: TreeNode, instance, sampDictAttr): 
    # initialize the predicted class and the correct class
    predict = tree.majority
    classIndex = list(sampDictAttr.values()).index("class")
    correct = instance[classIndex]
    # if the current node is a leaf node return the vals
    if tree.type == 'leaf':
        predict = tree.label
        return predict, correct
    # find the index of the attribute used to split the tree at the current node
    testindex = list(sampDictAttr.keys()).index(tree.sampAttr)
    # check if attribute is numerical, hence use the threshold to decide which subtree to go to
    if tree.dataType == "numerical":
        nexttree = tree.edge['<='] if instance[testindex] <= tree.thres else tree.edge['>']
    # else if categorical, use the value of the instance to decide which subtree to go to
    else:
        if instance[testindex] not in tree.edge:
            return predict, correct
        nexttree = tree.edge[instance[testindex]]
    # recursively call with val of next node
    return predictLabel(nexttree, instance, sampDictAttr)
#________________________________________________________________________________________________________________________________________________________________________