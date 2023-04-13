from utils import *
from decisionTree import *
from randomForest import *
import csv

def extractData(name:str, delimit:str, classN:str, categ:str):
    file = open("datasets/" + name, encoding = 'utf-8-sig')
    reader = csv.reader(file, delimiter = delimit)
    data = []
    for row in reader:
        data.append(row)
    fileCateg = {}
    for i in data[0]:
        fileCateg[i] = categ
    fileCateg[classN] = 'class'
    fileData = np.array(data[1:]).astype(float)
    return fileData, fileCateg

def extractCmcData():

    def extractFile(name:str, delimit:str):
        file = open("datasets/"+name, encoding='utf-8-sig')
        reader = csv.reader(file, delimiter=delimit)
        data = []
        for row in reader:
            data.append(row)
        return data
    
    cmc = extractFile('cmc.data', ',')
    cmcCateg = {"Wife's age":"numerical","Wife's education":"categorical",
    "Husband's education":"categorical","Number of children ever born":"numerical",
    "Wife's religion":"binary","Wife's now working?":"binary",
    "Husband's occupation":"categorical","Standard-of-living index":"categorical",
    "Media exposure":"binary","Contraceptive method used":"class"}
    cmcData = np.array(cmc).astype(int)
    return cmcData, cmcCateg

nParams = [1, 5, 10, 20, 30, 40, 50] 

def getResultsWine(data, dataCateg, k, maxDepth, minSize, minGain, impurity, bootstrapRatio, name, nImpurity):
    dataAccuracy, dataPrecision, dataRecall, dataF1 = [], [], [], []
    for n in nParams:
        lists = kFoldCrossValid(data, dataCateg, k, n, maxDepth, minSize, minGain, impurity, bootstrapRatio)[0]
        beta = 1
        acc0, pre0, rec0, fsc0 = meanEval(lists, 1, beta)
        acc1, pre1, rec1, fsc1 = meanEval(lists, 2, beta)
        acc2, pre2, rec2, fsc2 = meanEval(lists, 3, beta)
        acc, pre, rec, fsc = (acc0 + acc1 + acc2) / 3, (pre0 + pre1 + pre2) / 3, (rec0 + rec1 + rec2) / 3, (fsc0 + fsc1 + fsc2) / 3
        dataAccuracy.append(acc)
        dataPrecision.append(pre)
        dataRecall.append(rec)
        dataF1.append(fsc)
        printMetrics(acc, pre, rec, fsc, beta, n, f"{name} with {nImpurity}")
    return dataAccuracy, dataPrecision, dataRecall, dataF1

def getResultsHouse(data, dataCateg, k, maxDepth, minSize, minGain, impurity, bootstrapRatio, name, nImpurity):
    dataAccuracy, dataPrecision, dataRecall, dataF1 = [], [], [], []
    for n in nParams:
        lists = kFoldCrossValid(data, dataCateg, k, n, maxDepth, minSize, minGain, impurity, bootstrapRatio)[0]
        beta = 1
        acc0, pre0, rec0, fsc0 = meanEval(lists, 0, beta)
        acc1, pre1, rec1, fsc1 = meanEval(lists, 1, beta)
        acc, pre, rec, fsc = (acc0 + acc1) / 2, (pre0 + pre1) / 2, (rec0 + rec1) / 2, (fsc0 + fsc1) / 2
        dataAccuracy.append(acc)
        dataPrecision.append(pre)
        dataRecall.append(rec)
        dataF1.append(fsc)
        printMetrics(acc, pre, rec, fsc, beta, n, f"{name} with {nImpurity}")
    return dataAccuracy, dataPrecision, dataRecall, dataF1

def getResultsCancer(data, dataCateg, k, maxDepth, minSize, minGain, impurity, bootstrapRatio, name, nImpurity):
    dataAccuracy, dataPrecision, dataRecall, dataF1 = [], [], [], []
    for n in nParams:
        lists = kFoldCrossValid(data, dataCateg, k, n, maxDepth, minSize, minGain, impurity, bootstrapRatio)[0]
        beta = 1
        acc, pre, rec, fsc = meanEval(lists, 1, beta)
        dataAccuracy.append(acc)
        dataPrecision.append(pre)
        dataRecall.append(rec)
        dataF1.append(fsc)
        printMetrics(acc, pre, rec, fsc, beta, n, f"{name} with {nImpurity}")
    return dataAccuracy, dataPrecision, dataRecall, dataF1

def plotter(dataAccuracy, dataPrecision, dataRecall, dataF1, title, xlabel, ylabel):
    plt.plot(nParams, dataAccuracy, color='blue', label='Accuracy')
    plt.plot(nParams, dataPrecision, color='green', label='Precision')
    plt.plot(nParams, dataRecall, color='red', label='Recall')
    plt.plot(nParams, dataF1, color='purple', label='F-1 score')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.legend()
    plt.show()


if __name__=="__main__":
    cmcData,cmcCateg = extractCmcData()
    lists,acc = kFoldCrossValid(cmcData, cmcCateg, k=10, ntree=20, maxdepth=10, minimalsize=10, minimalgain=0.01, algortype='id3', bootstrapratio = 0.1)
    print(acc)
    print(lists)


# # House Votes Dataset
# houseAccuracy, housePrecision, houseRecall, houseF1 = getResults(housedata, housecategory, nParams, 10, 10, 5, 0.01, 'id3', 0.1, "House")

# def plotter2(dataAccuracy, dataPrecision, dataRecall, dataF1):
#     plt.plot(nParams, dataAccuracy, color='blue', label='Accuracy')
#     plt.plot(nParams, dataPrecision, color='green', label='Precision')
#     plt.plot(nParams, dataRecall, color='red', label='Recall')
#     plt.plot(nParams, dataF1, color='purple', label='F-1 score')

#     # Set the title and labels for the plot
#     plt.title('HouseVote with Gini index')
#     plt.xlabel('Number of trees in the forest')
#     plt.ylabel('Metric score')

#     # Add a legend to show which color represents which metric
#     plt.legend()

#     # Display the plot
#     plt.show()

# plotter2(houseAccuracy, housePrecision, houseRecall, houseF1)