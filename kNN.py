from numpy import *
import operator
import sys

# Use Python to generate and import the data
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0] # Get the size of first row

    diffMat = tile(inX, (dataSetSize,1)) - dataSet # Repeat matrix inX dataSetSize times by row and 1 time by column, then minues dataSet

    sqDiffMat = diffMat**2 # ^2 for each element in matrix

    sqDistances = sqDiffMat.sum(axis=1) # sum by row

    distances = sqDistances**0.5 # sqrt the value

    sortedDistIndicies = distances.argsort() # sort the value

    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]] # get the label of this index
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1 # count+1

    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True) # sort by the second element of items, that's why we use operator.itemgetter(1) here, it reverse the key and value.

    return sortedClassCount[0][0] # get top one element

def main():

    if len(sys.argv) == 5:
        try:
            groups = eval(sys.argv[1])
        except (NameError, SyntaxError):
            groups = sys.argv[1]
        try:
            labels = eval(sys.argv[2])
        except (NameError, SyntaxError):
            labels = sys.argv[2]
        try:
            topN = eval(sys.argv[3])
        except (NameError, SyntaxError):
            topN = sys.argv[3]
        try:
            testData = eval(sys.argv[4])
        except (NameError, SyntaxError):
            testData = sys.argv[4]
        if isinstance(groups, list) and isinstance(labels, list) and isinstance(topN, int) and isinstance(testData,list):
            groups = array(groups)
            if groups.ndim == 2 and groups.shape[0] == len(labels) and groups.shape[1] == len(testData):
                print("Result:", classify0(testData, groups, labels, topN))
            else:
                print("Matrix Shape ERROR\nUsage: python kNN.py DATA_LIST LABEL TOPN TESTDATA")
        else:
            print("Type Error\nUsage: python kNN.py DATA_LIST LABEL TOPN TESTDATA")
    else:
        print("Usage: python kNN.py DATA_LIST LABEL TOPN TESTDATA")
        print("Examle: DATA_LIST = [[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]], LABEL = ['A','A','B','B'], TOPN = 3")
        print("Input data is [1,1]: ")
        groups, labels = createDataSet()
        print("Build-in Result:",classify0([1,1],groups,labels,3))

if __name__ == "__main__":
    main()