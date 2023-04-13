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

#___________________________________________________________________________________________________________________________