from utils import *
from decisionTree import *

def bootStrap(data, ratio=0.1):
    nSamp = len(data)
    nKeep = round(nSamp * (1 - ratio))
    indices = np.random.choice(nSamp, size=nKeep, replace=False)
    reSampled = data[indices]
    nMissing = nSamp - nKeep
    missingIndices = np.random.choice(nKeep, size=nMissing, replace=True)
    missingData = reSampled[missingIndices]
    return np.concatenate([reSampled, missingData])

# Random Forest, plant a forest of n trees
def plantNForest(data, categDict, nTree=10, maxDepth=10, minSize=10, minGain=0.01, impurity='id3', bootstrapRatio = 0.1):
    forest = []
    for i in range(nTree):
        datause = bootStrap(data, bootstrapRatio)
        tree = decisionTreeForest(datause, categDict, impurity, maxDepth, minSize, minGain)
        forest.append(tree)
    return forest

def stratifiedKFold(data, categDict, k=10):

    classIndex = list(categDict.values()).index("class")
    classes, cnts = np.unique(data[:,classIndex], return_counts=True)
    nClass = len(classes)
    listClasses = [data[data[:,classIndex]==c] for c in classes]
    listClasses = [np.random.permutation(c) for c in listClasses]
    splittedList = [np.array_split(c, k) for c in listClasses]
    combinedList = [np.concatenate([splittedList[i][j] for i in range(nClass)]) for j in range(k)]
    return combinedList

# A complete k-fold cross validation
def kFoldCrossValid(data, categDict, k=10, nTree=10, maxDepth=5, minSize=10, minGain=0.01, impurity='id3', bootstrapRatio = 0.1):

    # Predict the class of a single instance
    def forestVoteCalc(forest, instance, categDict):
        votes = {}
        for tree in forest:
            predict, correct = predictLabel(tree, instance, categDict)
            if predict not in votes:
                votes[predict] = 1
            else:
                votes[predict] += 1
        return max(votes, key=votes.get), correct
    
    folded = stratifiedKFold(data, categDict, k)
    listOfNd = []
    accList = []
    for i in range(k):
        sampDataSet = folded[i]
        sampFolded = folded.copy()
        sampFolded.pop(i)
        dataSetTrain = np.vstack(sampFolded) 
        corrCount = 0
        forestTrain = plantNForest(dataSetTrain,categDict,nTree,maxDepth,minSize,minGain,impurity,bootstrapRatio)
        analysis = []
        for instance in sampDataSet:
            predict, correct = forestVoteCalc(forestTrain,instance,categDict)
            analysis.append([predict, correct])
            if predict == correct:
                corrCount += 1
        listOfNd.append(np.array(analysis))
        accList.append(corrCount/len(sampDataSet))
    acc = np.mean(accList)
    return listOfNd, acc
