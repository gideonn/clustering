# 1. store values
#  - Give each value its own cluster and add to cluster array
# 2. make a distance function to calculate the min distance (minD(a,b))of two clusters using PROXIMITY CLUSTER/DISTANCE MATRIX/EUCLIDEAN DISTANCE
# 3. remove from the array of cluster
# 4. merge them and add back to the cluster array

import numpy as np
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
    return data_set

# using our own data structure for te given set of genes. Converting the data into ids, cluster number and points.
class Gene(object):
    id = 0
    cluster = 0
    point = []

    def __init__(self, id, cluster, point):
        self.id = id
        self.cluster = cluster
        self.point = point

# creates object of type Gene using class gene
def make_gene(point):
    gene = Gene(int(point[0]),int(point[1]), point[2:])
    return gene

# returns eucledian distance between two points
def getDistance(x, y):
    distance = np.sqrt(np.sum((np.asarray(x) - np.asarray(y)) ** 2))
    return distance	

# a starting cluster is created using the data points in not including id or cluster number.
# the number of clusters here is same as number of data points.
def createClusterSet(data_set):
	cluster = []
	for data in data_set:
		data = data[2:]
		cluster.append(data)
	return cluster


# we check if a given cluster is a leaf of another cluster by checking the length of cluster array.
def leaf_present(cluster):
	return len(cluster) == 1

# if the cluster has a leaf we get the childern of that cluster
def generateChild(cluster):
    if leaf_present(cluster):
        raise TypeError("no children")
    else:
        return cluster[1]

# In an array/tuple of leaf and child we try to get just the points using this function. 
# Extraction of points
def extractVal(cluster):
    if leaf_present(cluster):
        return cluster # is already a 1-tuple containing value
    else:
        return [value
                for child in generateChild(cluster)
                for value in extractVal(child)]

#getting the minimum distance between two clusters using eucledian distance
def getClusterDistance(cluster1, cluster2):

	value1 = extractVal(cluster1)
	value2 = extractVal(cluster2)

	return min([getDistance(input1, input2)
                        for input1 in value1
                        for input2 in value2])

def execute_Agglo(data_set):
	initial_cluster = createClusterSet(data_set)
	#convert the inputs into the form of 1-tuple
	clusters = [(i,) for i in initial_cluster]

	# Run a loop for agglo. As there is one base cluster in agglo.
	# Loop should run until length of clusters is 1.
	while len(clusters) > 1:
		# making a tuple and getting distances between them and then getting the min and storing that tuple.
		# getting min distance between two clusters using key as the function to get min distance
		clusA, clusB = min([(cluster1, cluster2) for i, cluster1 in enumerate(clusters)
    		for cluster2 in clusters[:i]],
    		key=lambda (x, y): getClusterDistance(x, y))
		#remove the occuring values from cluster
		clusters = [cluster for cluster in clusters if cluster != clusA and cluster != clusB]
		# make a marged_cluster out of the values with the length of a given cluster and the values.
		merged_cluster = (len(clusters), [clusA, clusB])
		#append new values to cluster
		clusters.append(merged_cluster)

	#return the first index of cluster
	return clusters[0]

# getting the length/size of the merge of that cluster 
def get_merge_order(cluster):
    if leaf_present(cluster):
        return float('inf')
    else:
        return cluster[0] 

# After having one base cluster, the desired number of clusters can be generated using min_clusters.
def generate_clusters(base_cluster, num_clusters):
    # make a cluster that has only base_cluster
    clusters = [base_cluster]
    
    # Run the loop until we have the desired number of clusters.
    while len(clusters) < num_clusters:
        # choose the last-merged of our clusters
        next_cluster = min(clusters, key=get_merge_order)
        # remove it from the list
        clusters = [cluster for cluster in clusters if cluster != next_cluster]
        # and add its children to the list (i.e., unmerge it)
        clusters.extend(generateChild(next_cluster))
    # return the desired number of clusters
    return clusters

# returns all possible pairs of given data points
def getPairs(data_points):
    pairs = []
    for i in range(len(data_points)):
        for j in range(i + 1, len(data_points)):
            pairs.append([data_points[i], data_points[j]])
            pairs.append([data_points[j], data_points[i]])
    return pairs

# returns cluster matrix
def createClusterMatrix(data_set,all_clusters):
    Matrix = [[0 for x in range(len(data_set))] for y in range(len(data_set))]

    dict = {}
  
    for i in range(len(data_set)):
    	gene = make_gene(data_set[i])
    	for j in range(len(all_clusters)):
    		if gene.point in all_clusters[j]:
    			gene.cluster = j
        dict[i] = gene
    
    for i in range(len(data_set)-1):
    	for j in range(i+1,len(data_set)):
    		gene1 = dict[i]
    		gene2 = dict[j]

    		if gene1.cluster == gene2.cluster:
    			ids = [gene1.id, gene2.id]
    			pairs = getPairs(ids)
    			for p in pairs:
    				Matrix[p[0]-1][p[1]-1] = 1
    return Matrix , dict

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
    point = []
    for key in dic:
    	gene = dic[key]
    	point.append(gene.point)
    	c.append(str(gene.cluster))

    finalData = np.array(point)
    labels = np.array(c)
    pca = decomposition.PCA(n_components=2)
    pca.fit(finalData)
    finalData = np.mat(pca.transform(finalData))
    return finalData, labels


# plot graph
def plotGraph(filename, finalData,labels):
    #create dataframe and group based on labels
    df = pd.DataFrame(dict(x=np.asarray(finalData.T[0])[0], y=np.asarray(finalData.T[1])[0], label=labels))
    groups = df.groupby('label')
    fig, ax = plt.subplots()
    ax.margins(0.05)

    #plot all datapoints
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=5, label=name)

    ax.legend()
    ax.set_title('Algorithm: Agglomerate\n Input file: ' + filename)
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')

    plt.savefig('PCA_' + filename + ".png", dpi=300)
    plt.show()


def main():
    # getting the arguments from command-line and storing them for further use using optionparser
    optparser = OptionParser()
    optparser.add_option('-f', '--inputFile', dest='filename', help='filename of dataset',
                         default='iyer.txt')
    optparser.add_option('-k', '--number_of_cluster_points', dest='clustercount', help='number of clusters', default=5, type='int')
    (options, args) = optparser.parse_args()

    # storing the inputs into the variables
    file_path = options.filename
    k = options.clustercount

    # getting dataset from the loadDataSet function by giving file path
    data_set = loadDataSet(file_path)

    # executeAgglo(data_set,number_of_cluster_points)
    cluster = execute_Agglo(data_set)

    generated_clusters = generate_clusters(cluster, k)
    all_clusters = []

    for cluster in generated_clusters:
        all_clusters.append(extractVal(cluster))	

    # # create cluster matrix
    cluster_matrix, dict = createClusterMatrix(data_set,all_clusters)

    # #create ground truth matrix
    ground_truth_matrix = createGroundTruthMatrix(data_set)

    # #create incidence matrix
    incidence_matrix = createIncidenceMatrix(cluster_matrix,ground_truth_matrix)

    # #calculate rand coefficient
    randCoefficient = calculateRandCoefficient(incidence_matrix)

    # #calculate jaccard coefficient
    jaccardCoefficient = calculateJaccardCoefficient(incidence_matrix)

    print randCoefficient
    print jaccardCoefficient

    #perform PCA
    data, labels = PCA(dict)

    #plot Graph
    plotGraph(file_path.split('/')[-1],data,labels)

if __name__ == "__main__": main()