from evaluationMatrix import *
import numpy as np
import math
from collections import Counter

def same(attrCol):
    return all(attr == attrCol[0] for attr in attrCol)

def majority(attrCol):
    return np.argmax(np.bincount(attrCol.astype(int)))

def bestSeparate(dataSet, attrs:dict, impurity:str):

    def entropy(attributecol):
        values = Counter(attributecol).values()
        ent = 0
        for value in values:
            j = (value/len(attributecol))
            ent += -j*math.log2(j)
        return ent

    def gini(attributecol):
        values = Counter(attributecol).values()
        ginivalue = 1 - sum((value/len(attributecol))**2 for value in values)
        return ginivalue
    
    datasetCol = dataSet.T
    classIndex = list(attrs.values()).index("class")
    
    # Set original impurity measure
    if impurity == 'entropy':
        origImp = entropy(datasetCol[classIndex])
        smallestImp = origImp
    elif impurity == 'gini':
        origImp = gini(datasetCol[classIndex])
        smallestImp = origImp
    else:
        raise ValueError("impurity must be either 'entropy' or 'gini'")
    
    thresVal = -1

    i = 0
    bestAttr = {list(attrs.keys())[i]:attrs[list(attrs.keys())[i]]}
    currAttr = list(attrs.keys())[1:] if (classIndex == 0) else list(attrs.keys())[:classIndex]

    for attribute in currAttr:
        index = i + 1 if classIndex == 0 else i

        if attrs[attribute] == "categorical" or attrs[attribute] == "binary":
            listKeys = list(Counter(datasetCol[index]).keys())
            listCateg = [] 
            
            for key in listKeys:
                index_list = [idex for idex, element in enumerate(datasetCol[index]) if element == key]
                category = np.array(datasetCol[classIndex][index_list])
                listCateg.append(category)

            currImpurity = 0

            for category in listCateg:
                a = len(category) / len(datasetCol[index])
                if impurity == 'entropy':
                    currImpurity += a * entropy(category)
                elif impurity == 'gini':
                    currImpurity += a * gini(category)

            if currImpurity < smallestImp:
                smallestImp = currImpurity
                bestAttr = {attribute: attrs[attribute]}

        elif attrs[attribute] == "numerical":
            sortedDataset = datasetCol.T[datasetCol.T[:, index].argsort(kind='quicksort')].T
            currThres = (sortedDataset[index][1] + sortedDataset[index][0]) / 2
            j = 1
            while j < len(sortedDataset.T):
                currThres = (sortedDataset[index][j] + sortedDataset[index][j - 1]) / 2
                listCateg = [sortedDataset[classIndex][:j], sortedDataset[classIndex][j:]]
                currImpurity = 0

                for category in listCateg:
                    a = len(category) / len(datasetCol[index])
                    if impurity == 'entropy':
                        currImpurity += a * entropy(category)
                    elif impurity == 'gini':
                        currImpurity += a * gini(category)

                if currImpurity < smallestImp:
                    smallestImp = currImpurity
                    thresVal = currThres
                    bestAttr = {attribute: attrs[attribute]}
                j += 1
        i += 1

    gain = origImp - smallestImp
    return bestAttr, thresVal, gain
