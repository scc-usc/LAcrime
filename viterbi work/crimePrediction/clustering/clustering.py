"""
Author : Omkar Damle
date : 26th June 2017

This file takes a numpy data file as input. The input file contains crime locations. 
Using DBSCAN clustering, this code produces two numpy files, namely cluster_matrix and cluster_cells. 
The cluster information is used in linear regression prediction.
"""

import numpy as np

import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt

from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
import pandas as pd
import math
import sys
import matplotlib
matplotlib.rcParams.update({'font.size': 22})

dtCrime = np.load('../downTownCrimesLocations.npy')

print len(dtCrime)


rows = int(sys.argv[1])
cols = int(sys.argv[2])

#clustering paramter
# As epsilon increases, cluster size increases and number of clusters may decrease

epsilon = float(sys.argv[3])

# As min_samples increases, number of clusters will decrease
min_samples = int(sys.argv[4])

count = 0
lons = []
lats = []

crimeTime = []
for loc in dtCrime[0:1000]:
	if count%1000 == 0 :
		print count
	lons.append(loc[0])
	lats.append(loc[1])
	crimeTime.append(loc[2])
	count += 1


X = pd.DataFrame({'lat' : lats, 'lon': lons, 'timeAndDate' : crimeTime})

#plt.plot(lons,lats,'ro')
#plt.show()







def haversine(lonlat1, lonlat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lat1, lon1 = lonlat1
    lat2, lon2 = lonlat2
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r


#X = pd.read_csv('dbscan_test.csv')
distance_matrix = squareform(pdist(pd.DataFrame({'lat':X['lat'],'lon': X['lon']}), (lambda u,v: haversine(u,v))))


db = DBSCAN(eps=epsilon, min_samples=min_samples, metric='precomputed')  # using "precomputed" as recommended by @Anony-Mousse
y_db = db.fit_predict(distance_matrix)

X['cluster'] = y_db

clusters = pd.Series.unique(X['cluster'])
clusters.sort()
#print clusters
scatterPlots = [None for i in range(len(clusters))]
legends = [None for i in range(len(clusters))]
#colors = ['r', 'g', 'b', 'c', 'm', 'y','k']
count = 0

fig = plt.figure()

for clusterNo in clusters:
	temp = X.query('cluster == @clusterNo')
	scatterPlots[count] =  plt.scatter(temp['lon'], temp['lat'], c=np.random.rand(1,3))
	legends[count] = 'Cluster No: ' + str(clusterNo)
	count += 1

#print y_db
#plt.legend(scatterPlots, legends)
plt.xlabel('longitude')
plt.ylabel('latitude')

plt.title('DBSCAN clustering for following parameters: ' + str(rows) + 'x' + str(cols) + '_DTLA_' + str(epsilon) + '_' + str(min_samples))
fig.savefig('clusteringPlot.png')
plt.show()


"""
cluster grid cells
"""


#Downtown LA
# Lower Left -> 34.038811, -118.273534
# Upper Right -> 34.053781, -118.237727

#horizontal length = 3.27km
#vertical length = 1.74km

#Central LA
# Lower left -> 33.930351, -118.479460
#upper right -> 34.106495, -118.223809

ll_lat = 34.038811
ll_lon = -118.273534
ur_lat = 34.053781 
ur_lon = -118.237727

cluster_matrix = [[set() for x in range(cols)] for y in range(rows)]
crime_matrix = [[[] for x in range(cols)] for y in range(rows)]

#list = matrix[0][0]
#list.append(3)

height = math.fabs(ur_lat - ll_lat)/rows
width = math.fabs(ur_lon - ll_lon)/cols

#print height,width

count = 0
n = 0

for each in X.iterrows():
	#t = each[1]['Date Occurred']
	#print each[1]['Date Occurred']
	#print 'time'
	#print each[1]['timeAndDate']
	n += 1 

	if n%10000 == 0:
		print n
		#pass

	lon,lat = each[1]['lon'],each[1]['lat']
	#print lon,lat
	
	count += 1
	row_index = int(math.floor((ur_lat - lat)/height))
	col_index = int(math.floor((lon - ll_lon)/width))
	#print row_index,col_index


	myList = crime_matrix[row_index][col_index]
	myList.append(each[1]['timeAndDate'])

	cluster_matrix[row_index][col_index].add(each[1]['cluster'])

#print cluster_matrix


cluster_cells = [[] for i in range(len(clusters))]

for row in range(rows):
	for col in range(cols):
		tempList = list(cluster_matrix[row][col])
		for xx in range(len(tempList)):
			#print tempList[xx]	
			if tempList[xx]!=-1:
				cluster_cells[tempList[xx]].append((row,col))

#print cluster_cells
#plt.show()

#np.save('crime_matrix_' + str(rows) + 'x' + str(cols) + '_DTLA_' + str(epsilon) + '_' + str(min_samples) ,crime_matrix)
np.save('cluster_matrix' + str(rows) + 'x' + str(cols) + '_DTLA_' + str(epsilon) + '_' + str(min_samples), cluster_matrix)
np.save('cluster_cells' + str(rows) + 'x' + str(cols) + '_DTLA_' + str(epsilon) + '_' + str(min_samples), cluster_cells)
