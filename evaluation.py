from decisionTree import *
from randomForest import *
from evaluation import *
import matplotlib.pyplot as plt
import csv

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
#________________________________________________________________________________________________________________________________________________________________________

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

def evalResults(results, corrResult, beta=1):
    accList, preList, recList, f1List = [], [], [], []
    for res in results:
        truePosi = sum(p == a == corrResult for p, a in res)
        trueNega = sum(p == a != corrResult for p, a in res)
        falsePosi = sum(p == corrResult and a != corrResult for p, a in res)
        falseNega = sum(p != corrResult and a == corrResult for p, a in res)
        accList.append(accuracy(truePosi, trueNega, falsePosi, falseNega))
        preList.append(precision(truePosi, falsePosi))
        recList.append(recall(truePosi, falseNega))
        f1List.append(fscore(truePosi, falsePosi, falseNega, beta))
    return accList, preList, recList, f1List

def meanEval(results, corrResult, beta=1):
    accuarcylists, precisionlists, recalllists, fscorelists = evalResults(results, corrResult, beta)
    return sum(accuarcylists)/len(accuarcylists), sum(precisionlists)/len(precisionlists), sum(recalllists)/len(recalllists), sum(fscorelists)/len(fscorelists)

def printMetrics(acc, pre, rec, fsc, beta, nvalue, title):
    acc, pre, rec, fsc = round(acc, 3), round(pre, 3), round(rec, 3), round(fsc, 3)
    print(f"{nvalue} Trees Random Forest of {title}:")
    print(f"Accuracy: {acc}")
    print(f"Precision: {pre}")
    print(f"Recall: {rec}")
    print(f"F-Score(beta={beta}): {fsc}")

nParams = [1, 5, 10, 20, 30, 40, 50] 
#________________________________________________________________________________________________________________________________________________________________________

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

#________________________________________________________________________________________________________________________________________________________________________