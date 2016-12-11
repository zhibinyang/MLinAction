from numpy import *
import operator

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

if __name__ == "__main__":
    groups, labels = createDataSet()
    print classify0([1,1],groups,labels,3)
