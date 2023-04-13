from evaluation import *
from decisionTree import *
from randomForest import *

housedata, housecategory = extractData('hw3_house_votes_84.csv', ',', "class", 'categorical')
winedata, winecategory = extractData('hw3_wine.csv', '\t', "# class", 'numerical')
cancerdata, cancercategory = extractData('hw3_cancer.csv', '\t', "Class", "numerical")

# WINE DATASET ID3
wineaccuracy, wineprecision, winerecall, winef1 = getResultsWine(winedata, winecategory, 10, 10, 5, 0.01, 'id3', 0.1, "Wine Dataset", "ID3")
plotter(wineaccuracy, wineprecision, winerecall, winef1, "n Vs Metrics for Wine Dataset with ID3", "n", "Score")

# HOUSE DATASET ID3
houseaccuracy, houseprecision, houserecall, housef1 = getResultsHouse(housedata, housecategory, 10, 10, 5, 0.01, 'id3', 0.1, "House Dataset", "ID3")
plotter(houseaccuracy, houseprecision, houserecall, housef1, "n Vs Metrics for House Dataset with ID3", "n", "Score")

# WINE DATASET GINI
wineaccuracy, wineprecision, winerecall, winef1 = getResultsWine(winedata, winecategory, 10, 10, 5, 0.01, 'cart', 0.1, "Wine Dataset", "Gini")
plotter(wineaccuracy, wineprecision, winerecall, winef1, "n Vs Metrics for Wine Dataset with Gini", "n", "Score")

# HOUSE DATASET GINI
houseaccuracy, houseprecision, houserecall, housef1 = getResultsHouse(housedata, housecategory, 10, 10, 5, 0.01, 'cart', 0.1, "House Dataset", "Gini")
plotter(houseaccuracy, houseprecision, houserecall, housef1, "n Vs Metrics for House Dataset with Gini", "n", "Score")

# CANCER DATASET ID3
canceraccuracy, cancerprecision, cancerrecall, cancerf1 = getResultsCancer(cancerdata, cancercategory, 10, 10, 5, 0.01, 'id3', 0.1, 'Cancer Dataset', 'ID3')
plotter(canceraccuracy, cancerprecision, cancerrecall, cancerf1, "n Vs Metrics for Cancer Dataset with ID3", "n", "Score")

# CANCER DATASET GINI
canceraccuracy, cancerprecision, cancerrecall, cancerf1 = getResultsCancer(cancerdata, cancercategory, 10, 10, 5, 0.01, 'cart', 0.1, 'Cancer Dataset', 'Gini')
plotter(canceraccuracy, cancerprecision, cancerrecall, cancerf1, "n Vs Metrics for Cancer Dataset with Gini", "n", "Score")