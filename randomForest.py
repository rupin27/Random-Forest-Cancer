from utils import *
from decisionTree import *

# random Forest algorithm to plant a forest of n trees
def plantNForest(data, categDict, nTree=10, maxDepth=10, minSize=10, minGain=0.01, impurity='id3', bootstrapRatio = 0.1):
    # bootstrapping helper
    def bootStrap(data, ratio=0.1):
        nSamp = len(data)
        nKeep = round(nSamp * (1 - ratio))
        # select random samples without replacing
        indices = np.random.choice(nSamp, size=nKeep, replace=False)
        reSampled = data[indices]
        # repeating the same process
        nMissing = nSamp - nKeep
        missingIndices = np.random.choice(nKeep, size=nMissing, replace=True)
        missingData = reSampled[missingIndices]
        # returning the combined list
        return np.concatenate([reSampled, missingData])
    
    forest = []
    # here we plant the ntree trees into the forest
    for i in range(nTree):
        # get the bootstrapped data
        datause = bootStrap(data, bootstrapRatio)
        # build
        tree = decisionTreeForest(datause, categDict, impurity, maxDepth, minSize, minGain)
        forest.append(tree)
    return forest

# stratified k-fold cross validation
def stratifiedKFold(data, categDict, k=10):

    classIndex = list(categDict.values()).index("class")
    # getting all the unique classes
    classes = np.unique(data[:,classIndex])
    nClass = len(classes)
    # splitting the data based on class
    listClasses = [data[data[:,classIndex]==c] for c in classes]
    # into k folds
    listClasses = [np.random.permutation(c) for c in listClasses]
    splittedList = [np.array_split(c, k) for c in listClasses]
    # combing for the whole dataset
    combinedList = [np.concatenate([splittedList[i][j] for i in range(nClass)]) for j in range(k)]
    return combinedList
#________________________________________________________________________________________________________________________________________________________________________

# k-fold cross validation
def kFoldCrossValid(data, categDict, k=10, nTree=10, maxDepth=5, minSize=10, minGain=0.01, impurity='id3', bootstrapRatio = 0.1):

    # predict the class of a instance using rf
    def forestVoteCalc(forest, instance, categDict):
        votes = {}
        for tree in forest:
            #predicting the class
            predict, correct = predictLabel(tree, instance, categDict)
            if predict not in votes:
                votes[predict] = 1
            else:
                votes[predict] += 1
        # class with most votes
        return max(votes, key=votes.get), correct
    # split the dataset into k folds, and then test and train the rf k times
    folded = stratifiedKFold(data, categDict, k)
    listOfNd = []
    accList = []
    for i in range(k):
        # test instances
        sampDataSet = folded[i]
        # folds and then remove the current fold
        sampFolded = folded.copy()
        sampFolded.pop(i)
        # training data for current fold
        dataSetTrain = np.vstack(sampFolded) 
        # keep the count for correctly predicted instances
        corrCount = 0
        # planting the ntrees 
        forestTrain = plantNForest(dataSetTrain, categDict, nTree, maxDepth, minSize, minGain, impurity, bootstrapRatio)
        result = []
        for instance in sampDataSet:
            predict, correct = forestVoteCalc(forestTrain,instance,categDict)
            result.append([predict, correct])
            if predict == correct:
                corrCount += 1
        # calculating the results
        listOfNd.append(np.array(result))
        accList.append(corrCount/len(sampDataSet))
    # taking a mean and returning the result
    acc = np.mean(accList)
    return listOfNd, acc
#________________________________________________________________________________________________________________________________________________________________________
