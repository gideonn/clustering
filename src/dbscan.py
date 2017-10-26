import numpy
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

class Gene(object):
    id = 0
    cluster = 0
    point = []
    status = 'Unvisited'
    IsClusterAssigned = False


    def __init__(self, id, cluster, point):
        self.id = id
        self.cluster = cluster
        self.point = point

# creates object of type Gene
def make_gene(point):
    gene = Gene(int(point[0]),int(point[1]), point[2:])
    return gene

# returns eucledian distance between two points
def getDistance(x, y):
    distance = numpy.sqrt(numpy.sum((numpy.asarray(x) - numpy.asarray(y)) ** 2))
    return distance

# return Neighbouring points of a given point
def regionQuery(gene, eps, GenesData):
    Neighbourpoints = []
    for genepoint in GenesData:
        distance = getDistance(gene.point, genepoint.point)
        if distance<=eps:
            Neighbourpoints.append(genepoint)
    return Neighbourpoints

# expand a given cluster
def expandCluster(gene, NeighbourPoints, C, eps, MinPts, GenesData, dict):
    gene.IsClusterAssigned = True
    if C in dict:
        dict[C].append(gene)
    else:
        dict[C] = [gene]
    for point in NeighbourPoints:
        if point.status != 'Visited':
            point.status = 'Visited'
            # NeighbourPoints2 are the neighbours of NeighbourPoints
            NeighbourPoints2 = regionQuery(point, eps, GenesData)
            if len(NeighbourPoints2) >= MinPts:
                NeighbourPoints.extend(list(set(NeighbourPoints2)));
        if point.IsClusterAssigned == False:
            point.IsClusterAssigned = True
            dict[C].append(point)
    return dict

# DBScan
def DBScan(GenesData, eps, MinPts):
    dict = {}
    C = 0
    for gene in GenesData:
        if gene.status == 'Unvisited':
            gene.status = 'Visited'
            NeighbourPoints = regionQuery(gene , eps, GenesData)
            if len(NeighbourPoints)<MinPts:
                gene.status = 'Noise'
            else:
                C +=1
                dict = expandCluster(gene, NeighbourPoints, C, eps, MinPts, GenesData, dict)
    return dict

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
    ax.set_title('Algorithm: DBScan\n Input file: ' + filename)
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')

    plt.savefig('PCA_' + filename + ".png", dpi=300)
    plt.show()

def main():
    # getting the arguments from command-line and storing them for further use using optionparser
    optparser = OptionParser()
    optparser.add_option('-f', '--inputFile', dest='filename', help='filename of dataset',
                         default='C:/Users/divya/OneDrive/Documents/601/p2/cho.txt')
    #iyer =  1.149259, 2 = 11 clusters
    optparser.add_option('-e', '--eps', dest='eps', help='radius', default=1.03, type='int')
    optparser.add_option('-m', '--min_points', dest='min_points', help='minimum number of points', default=4, type='int')
    (options, args) = optparser.parse_args()

    # storing the inputs into the variables
    file_path = options.filename
    eps = options.eps
    min_points = options.min_points
    # getting dataset from the loadDataSet function by giving file path
    data_set = loadDataSet(file_path);

    # modify data
    GenesData = []
    for j in range(0, len(data_set)):
        gene = make_gene(data_set[j])
        GenesData.append(gene)

    #execute DBSCAN
    dict = DBScan(GenesData, eps, min_points)
    #print dict

    # create cluster matrix
    cluster_matrix = createClusterMatrix(len(data_set), dict)

    # create ground truth matrix
    ground_truth_matrix = createGroundTruthMatrix(data_set)

    # create incidence matrix
    incidence_matrix = createIncidenceMatrix(cluster_matrix, ground_truth_matrix)

    # calculate rand coefficient
    randCoefficient = calculateRandCoefficient(incidence_matrix)

    # calculate jaccard coefficient
    jaccardCoefficient = calculateJaccardCoefficient(incidence_matrix)

    print randCoefficient
    print jaccardCoefficient

    # perform PCA
    dataForPCA = {}
    for key in dict:
        list = []
        for gene in dict[key]:
            list.append(gene.point)
        dataForPCA[key] = list
        # print dataForPCA

    data, labels = PCA(dataForPCA)

    # plot Graph
    plotGraph(file_path.split('/')[-1], data, labels)

if __name__ == "__main__": main()