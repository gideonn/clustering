import numpy
import random
from optparse import OptionParser
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import decomposition

# loading dataSet
def loadDataSet(filePath):
    data_set = []
    with open(filePath) as file:
        for line in file:
            items = line.split('\t')
            res_line = [float(val) for i, val in enumerate(items[:-1])]
            last_item = float(items[-1].replace('\n', ''))
            res_line.append(last_item)
            data_set.append((res_line))
    #print data_set
    return data_set

# returns eucledian distance between two points
def getDistance(x, y):
    distance = numpy.sqrt(numpy.sum((numpy.asarray(x) - numpy.asarray(y)) ** 2))
    return distance

class Gene(object):
    id = 0
    cluster = 0
    point = []

    def __init__(self, id, cluster, point):
        self.id = id
        self.cluster = cluster
        self.point = point

# creates object of type Gene
def make_gene(point):
    gene = Gene(int(point[0]),int(point[1]), point[2:])
    return gene

# returns points under each cluster
def getClusterAssignment(data_set,clusters):
    distance = []
    dict = {}
    for j in range(0, len(data_set)):
        for i in range(0, len(clusters)):
            distance.append(getDistance(data_set[j][2:], clusters[i]))
        key = numpy.asarray(distance).argmin()
        gene = make_gene(data_set[j])
        if key in dict:
            dict[key].append(gene)
        else:
            dict[key] = [gene]
        distance = []
    return dict

# returns new clusters
def getNewClusters(data_set,clusters):
    prevCluster = clusters[:]
    dict = getClusterAssignment(data_set, clusters)
    for key in dict.keys():
        genes = dict[key]
        points = [gene.point for gene in genes]
        clusters[key] = numpy.mean(points, axis=0).tolist()
    return clusters, prevCluster, dict

# compares if two clusters are same
def compareClusters(prevCluster, clusters):
    for i in range(0,len(prevCluster)):
        if clusters[i] == prevCluster[i]:
            continue
        else:
            return False
    return True

# returns cluster with specific gene ids
def getCluster(data_set, geneIdList):
    geneData = []
    for j in range(0, len(data_set)):
        gene = make_gene(data_set[j])
        if gene.id in geneIdList:
            geneData.append(gene.point)
    return geneData

# Kmeans
def executeKMeans(data_set, number_of_cluster_points, number_of_iterations, geneIdList):
    #random_choice = random.sample(data_set, number_of_cluster_points)
    #clusters = [row[2:] for row in random_choice]
    clusters = getCluster(data_set, geneIdList)
    clusters, prevCluster, dict = getNewClusters(data_set,clusters)
    iter = number_of_iterations
    #while not compareClusters(prevCluster, clusters):
    while iter>0:
        iter-=1
        clusters, prevCluster, dict = getNewClusters(data_set,clusters)
    return clusters, prevCluster, dict

# returns all possible pairs of given data points
def getPairs(data_points):
    pairs = []
    for i in range(len(data_points)):
        for j in range(i + 1, len(data_points)):
            pairs.append([data_points[i], data_points[j]])
            pairs.append([data_points[j], data_points[i]])
    return pairs

# returns cluster matrix
def createClusterMatrix(data_set_length,dict):
    Matrix = [[0 for x in range(data_set_length)] for y in range(data_set_length)]
    for key in dict.keys():
        genes = dict[key]
        ids = [gene.id for gene in genes]
        pairs = getPairs(ids)
        for p in pairs:
            Matrix[p[0]-1][p[1]-1] = 1
    return Matrix

# returns ground truth matrix
def createGroundTruthMatrix(data_set):
    Matrix = [[0 for x in range(len(data_set))] for y in range(len(data_set))]
    dict = {}
    for j in range(0, len(data_set)):
        gene = make_gene(data_set[j])
        key = gene.cluster
        if key in dict:
            dict[key].append(gene)
        else:
            dict[key] = [gene]

    for key in dict.keys():
        genes = dict[key]
        ids = [gene.id for gene in genes]
        pairs = getPairs(ids)
        for p in pairs:
            Matrix[p[0]-1][p[1]-1] = 1
    return Matrix

# returns incidence matrix
def createIncidenceMatrix(cluster_matrix, ground_truth_matrix):
    Matrix = [[0 for x in range(2)] for y in range(2)]
    for i in range(len(cluster_matrix)):
        for j in range(len(cluster_matrix)):
            if cluster_matrix[i][j] == ground_truth_matrix[i][j]:
                if cluster_matrix[i][j] == 1:
                    Matrix[0][0] += 1 #same cluster,same cluster
                else:
                    Matrix[1][1] += 1 #different cluster, different cluster
            else:
                if cluster_matrix[i][j] == 1:
                    Matrix[0][1] += 1 #different cluster, same cluster
                else:
                    Matrix[1][0] += 1 #same cluster, different cluster
    #print Matrix
    return Matrix

# returns rand coefficient
def calculateRandCoefficient(incidence_matrix):
    num = float(incidence_matrix[0][0] + incidence_matrix[1][1])
    den = float(incidence_matrix[0][0] + incidence_matrix[1][1] + incidence_matrix[0][1] + incidence_matrix[1][0])
    randCoefficient = num/den
    return randCoefficient

#returns Jaccard coefficient
def calculateJaccardCoefficient(incidence_matrix):
    num = float(incidence_matrix[0][0])
    den = float(incidence_matrix[0][0] + incidence_matrix[0][1] + incidence_matrix[1][0])
    jaccardCoefficient = num/den
    return jaccardCoefficient

# perform PCA
def PCA(dic):
    c = []
    data = []
    for key in dic:
        for i in range(len(dic[key])):
            c.append(key)
            data.append(dic[key][i])
    finalData = numpy.array(data)
    labels = numpy.array(c)
    pca = decomposition.PCA(n_components=2)
    pca.fit(finalData)
    finalData = numpy.mat(pca.transform(finalData))
    return finalData, labels

# plot graph
def plotGraph(filename, finalData,labels):
    #create dataframe and group based on labels
    df = pd.DataFrame(dict(x=numpy.asarray(finalData.T[0])[0], y=numpy.asarray(finalData.T[1])[0], label=labels))
    groups = df.groupby('label')
    fig, ax = plt.subplots()
    ax.margins(0.05)

    #plot all datapoints
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=5, label=name)

    ax.legend()
    ax.set_title('Algorithm: KMeans\n Input file: ' + filename)
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')

    plt.savefig('PCA_' + filename + ".png", dpi=300)
    plt.show()

def main():
    # getting the arguments from command-line and storing them for further use using optionparser
    optparser = OptionParser()
    optparser.add_option('-f', '--inputFile', dest='filename', help='filename of dataset',
                         default='C:/Users/divya/OneDrive/Documents/601/p2/cho.txt')
    optparser.add_option('-c', '--number_of_cluster_points', dest='clustercount', help='number of clusters', default=3, type='int')
    optparser.add_option('-i', '--number_of_iterations', dest='iterations', help='number of iterations', default=10, type='int')
    (options, args) = optparser.parse_args()

    # storing the inputs into the variables
    file_path = options.filename
    number_of_cluster_points = options.clustercount
    number_of_iterations = options.iterations

    # getting dataset from the loadDataSet function by giving file path
    data_set = loadDataSet(file_path);

    geneIdList = [3,5,9]
    # executing Kmeans algorithm
    clusters, prevCluster, dict = executeKMeans(data_set, number_of_cluster_points, number_of_iterations, geneIdList)
    dataForPCA = {}
    for key in dict:
        list =[]
        for gene in dict[key]:
            list.append(gene.point)
        dataForPCA[key] = list
    #print dataForPCA

    # create cluster matrix
    cluster_matrix = createClusterMatrix(len(data_set), dict)

    #create ground truth matrix
    ground_truth_matrix = createGroundTruthMatrix(data_set)

    #create incidence matrix
    incidence_matrix = createIncidenceMatrix(cluster_matrix,ground_truth_matrix)

    #calculate rand coefficient
    randCoefficient = calculateRandCoefficient(incidence_matrix)

    #calculate jaccard coefficient
    jaccardCoefficient = calculateJaccardCoefficient(incidence_matrix)

    print randCoefficient
    print jaccardCoefficient

    #perform PCA
    data, labels = PCA(dataForPCA)

    #plot Graph
    plotGraph(file_path.split('/')[-1],data,labels)


if __name__ == "__main__": main()