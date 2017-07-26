"""
Author : Omkar Damle
Date: 29th June 2017

Naive implementation of a grid clustering algorithm and using the clusters for prediction
"""

import numpy as np
import sys

def insideGrid(rows,cols,cellRow, cellCol):
	if cellRow<0 or cellCol<0 or cellRow>=rows or cellCol>=cols:
		return False
	return True

def printMatrix(m):
	r = len(m)
	c = len(m[0])

	for i in range(r):
		for j in range(c):
			print m[i][j],
		print('')	










from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
from sklearn.model_selection import KFold
import pickle
from scipy import integrate
#import pandas as pd

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

import scipy.stats as sts

from pylab import *

def smoothen(array, window):
	#print array
	arr1 = [(1.0/window) for i in range(window)]
	smooth_array = np.convolve(array, arr1, mode = 'valid')
	#print smooth_array
	return smooth_array


row = int(sys.argv[1])
col = int(sys.argv[2])
time_period = int(sys.argv[3])
lookback = int(sys.argv[4])
#print row,col,time_period,lookback

countThreshold = int(sys.argv[5])
print('count threshold=' + str(countThreshold))


#test
#arr = [1,2,3,4,5,6,7,8]
#print smoothen(arr,3)
#raw_input('')

matrix = np.load('matrix_' + str(time_period) + 'h_' + str(row) + 'x' + str(col)+ '_DTLA_14to16.npy')


smoothing_window = 3

matrix_smooth = [[[] for x in range(col)] for y in range(row)]
matrix_shortened = [[[] for x in range(col)] for y in range(row)]

#smoothen the data
for i in range(row):
	for j in range(col):
		timeSeries = matrix[i][j]
		matrix_smooth[i][j] = smoothen(timeSeries,smoothing_window)
		#print matrix[i][j]

		#remove first and last element of original 
		if smoothing_window!=3:
			raise ValueError('Smoothing window is not 3! Make matrix_smooth and matrix elements of same length')

		matrix_shortened[i][j] = matrix[i][j][1:-1]

		#print len(matrix_smooth[i][j])
		#print len(matrix_shortened[i][j])

		#raw_input('')



print('FYI - Smoothing is done to data.')


len_data = len(matrix_smooth[0][0])
no_splits = 5

kf = KFold(n_splits=no_splits)

n_points = 101 #required for AUC calculation

#kf_sum_perc_crime__grid_clustering_cap = [0 for i in range(n_points)]
kf_sum_AUC = 0

#declared here so that they can be averaged over the splits
mean_sq_error_grid_clustering = 0
norm = 0

#print kf.split(np.zeros(len_data,2))


#In order to ensure uniform splitting
#quotient = len_data/no_splits
#len_data = quotient*no_splits
clusterSizeSum = 0
splitNo = 1

#mean squared error for grid clustering
mse_c = 0
count_mse = 0


mse_dt = 0
count_dt = 0
#time series of crimes which have happened in enitre DT area
dtTimeSeries = []

dtTimeSeriesActual = []

#WAPE(weighted absolute percentage error) variables
wape_diff_sum_dt = 0	#numerator for WAPE
wape_actual_sum_dt = 0 #denominator for WAPE

wape_diff_sum_cl = 0	#numerator for WAPE
wape_actual_sum_cl = 0 #denominator for WAPE




mse_lr = 0
count_mse_lr = 0

wape_diff_sum_lr = 0
wape_actual_sum_lr = 0



for t in range(len(matrix_smooth[0][0])):
	sum = 0
	sum_actual = 0
	for r in range(row):
		for c in range(col):
			sum += matrix_smooth[r][c][t]
			sum_actual += matrix_shortened[r][c][t]
	dtTimeSeries.append(sum)		
	dtTimeSeriesActual.append(sum_actual)


#This matrix is used to calculate the resource allocation metric
#The columns in this matrix are : split_no,timeInstant, prediction_list, actual_list

matrix_resource_metric = []

dict_area = {}

#units - sq km
area_entire_grid = 5.69
no_cells = row*col
area_cell = 5.69/no_cells
#area_list = [ area_cell for i in range(no_cells)]



for train_index,test_index in kf.split([[0 for i in range(2)] for j in range(len_data)]):

	#required for resource allocation metric
	area_updated = False
	
	"""
	Predicting for downtown as a whole	
	"""

	dtModel = LinearRegression()

	indep_vars_dt = []
	dep_var_dt = []


	for timeIter in train_index:
		#restrict the independent and dependent variables to training set only
		if timeIter+lookback in test_index:
			continue

		#reached the end	
		if timeIter+lookback == len_data:
			break	

		indep_vars_dt.append(dtTimeSeries[timeIter:(timeIter + lookback )])	
		dep_var_dt.append(dtTimeSeries[timeIter+lookback])

	dtModel.fit(indep_vars_dt, dep_var_dt)

	features_dt = []
	for timeIter in test_index:
		if (timeIter+lookback) not in test_index:
			break 
		features_dt.append(dtTimeSeriesActual[timeIter:(timeIter+lookback)])	
			
	pred_dt = dtModel.predict(features_dt)	
	#mse_dt = 0

	for x in range(len(pred_dt)):
	#	mse_dt += math.pow((pred_dt[x] - dtTimeSeries[x + test_index[0] + lookback]),2)
		mse_dt += math.pow(pred_dt[x] - dtTimeSeriesActual[x + test_index[0] + lookback],2)
		count_dt += 1

		wape_diff_sum_dt += abs(pred_dt[x] - dtTimeSeriesActual[x + test_index[0] + lookback])
		wape_actual_sum_dt += dtTimeSeriesActual[x + test_index[0] + lookback]


	#print('The test indices are:' + str(test_index[0]) + " to " + str(test_index[-1]))
	#print('The train indices are:' + str(train_index))

	
	crime_count_matrix = [[0 for j in range(col)] for i in range(row)]

	#fill the crime count matrix using training data and do the cliustering

	for i in range(row):
		for j in range(col):
			crime_count_matrix[i][j] = np.sum(matrix_smooth[i][j][train_index])

	#printMatrix(crime_count_matrix)		
	
	

	list1 = []
	for i in range(row):
		for j in range(col):
			count = crime_count_matrix[i][j]
			list1.append((i,j,count))

	#sort based on count
	list1.sort(key = lambda tuple:tuple[2],reverse = True)

	clusters = []

	#countThreshold = int(raw_input('Input the threshold count'))
	#countThreshold = 500
	booleanMatrix = [[True for i in range(col)] for j in range(row)]
	#make False when this cell is assigned to a cluster

	clusterMatrix = [[-1 for i in range(col)] for j in range(row)]
	clusterNo = 0

	for each in list1:
		flag = booleanMatrix[each[0]][each[1]]

		if flag == False:
			continue

		row1 = each[0]
		col1 = each[1]

		clusterSet = set()
		clusterSet.add((row1,col1))
		clusterMatrix[row1][col1] = clusterNo
		neighbors = set()

		for i in range(row1-1,row1+2):
			for j in range(col1-1,col1+2):

				if i == row1 and j == col1:
					continue

				if insideGrid(row,col,i,j) and booleanMatrix[i][j] == True:
					neighbors.add((i,j,crime_count_matrix[i][j]))

		count = each[2]

		while count<countThreshold:
			nList = list(neighbors)

			if len(nList) == 0:
				break

			nList.sort(key= lambda tuple:tuple[2],reverse = True)		

			count+= nList[0][2]

			tempRow = nList[0][0]
			tempCol = nList[0][1]
			clusterSet.add((tempRow,tempCol))
			booleanMatrix[tempRow][tempCol] = False
			clusterMatrix[tempRow][tempCol] = clusterNo

			neighbors.remove((tempRow,tempCol,crime_count_matrix[tempRow][tempCol]))

			for i in range(tempRow-1,tempRow+2):
				for j in range(tempCol-1,tempCol+2):
					if insideGrid(row,col,i,j) and booleanMatrix[i][j] == True:
						neighbors.add((i,j,crime_count_matrix[i][j]))

		#add cluster to cluster list!!!!!!!!!!!
		clusters.append(clusterSet)


		booleanMatrix[each[0]][each[1]] = False
		clusterNo += 1

	#printMatrix(clusterMatrix)

	#print('no of clusters : ' + str(len(clusters)))
	
	clusterSizeSum += len(clusters)
	#printMatrix(booleanMatrix)

	cluster_time_series = [[0 for yy in range(len_data)] for xx in range(len(clusters))]
	cluster_time_series_actual = [[0 for yy in range(len_data)] for xx in range(len(clusters))]

	pred_list_grid_clustering = [[0 for time in range(len(test_index) - lookback)] for c in range(len(clusters))]

	reg_mat_grid_clustering = [LinearRegression() for c in range(len(clusters))]

	count = 0
	for c in clusters:
		clusterSet = c

		temp_time_series = [0 for yy in range(len_data)]
		temp_time_series_actual = [0 for yy in range(len_data)]

		for cell in clusterSet:
			temp_time_series = np.add(temp_time_series,matrix_smooth[cell[0]][cell[1]])
			temp_time_series_actual = np.add(temp_time_series_actual, matrix_shortened[cell[0]][cell[1]])

		cluster_time_series[count] = temp_time_series	
		cluster_time_series_actual[count] = temp_time_series_actual

		#print cluster_time_series
		count+=1

	#raw_input('')


	for m in range(len(clusters)):
			#print('i am in cell : ' + str(i) + ',' + str(j))
		indep_var_mat = []
		dep_var_col = []

		timeIter=0
		cellList = cluster_time_series[m]

		#print cellList
		#raw_input('')

		cellList_actual = cluster_time_series_actual[m]

		for timeIter in train_index:
			#restrict the independent and dependent variables to training set only
			if timeIter+lookback in test_index:
				continue

			#reached the end	
			if timeIter+lookback == len_data:
				break	

#			while (k+lookback) < len_training_data:
			indep_var_mat.append(cellList[timeIter:(timeIter + lookback )])
			
			dep_var_col.append(cellList[timeIter+lookback])
			timeIter += 1

		reg_mat_grid_clustering[m].fit(indep_var_mat,dep_var_col)


		features_mat = []

		#while (k+lookback) < len_data:
		for timeIter in test_index:

			if (timeIter+lookback) not in test_index:
				break 

			features_mat.append(cellList_actual[timeIter:(timeIter+lookback)])

			timeIter +=1

		pred_list1 = reg_mat_grid_clustering[m].predict(features_mat)	

		for xx in range(len(pred_list1)):
			pred_list_grid_clustering[m][xx] = pred_list1[xx]


	"""
	let us calculate the AUC-ROC metric
	"""

	#raw_input('waiting for your permission')

	n_points = len(clusters)

	AUC = 0

	#in order to average across time instants
	perc_crime_cap_avg = [0 for i in range(n_points)]

	timeLength = len(pred_list_grid_clustering[0])

	pred_matrix_LR = np.load('pred_matrix_split_' + str(splitNo) + '.npy')	

	for timeInstant in range(timeLength):

		perc_crime_cap = [0 for i in range(n_points)]

		
		perc_area = [0 for i in range(n_points)]

		densities = []

		for clusterNo in range(n_points):
			density = float(pred_list_grid_clustering[clusterNo][timeInstant])/len(clusters[clusterNo])
			densities.append((density,clusterNo))

		densities.sort(reverse=True, key=lambda tuple1:tuple1[0])	

		total_no_cells = row*col
		running_sum_cells = 0

		total_crime = 0
		pred_sum = 0

		#list for all cells at a particular time instant
		predictionList = []
		actualList = []

		if area_updated == False:
			area_list = []

		for yy in range(n_points):


			currentClusterNo = densities[yy][1]
			running_sum_cells += len(clusters[currentClusterNo])
			perc_area[yy] += float(running_sum_cells)/total_no_cells

			#cluster prediction
			yc = pred_list_grid_clustering[currentClusterNo][timeInstant]

			#cluster reality
			xc = 0

			#LR prediction sum
			lr_pred = 0

			for cell1 in clusters[currentClusterNo]:
				tempRow = cell1[0]
				tempCol = cell1[1]
				pred_sum += matrix_smooth[tempRow][tempCol][timeInstant + test_index[0] + lookback]
				total_crime += pred_sum
				xc += matrix_shortened[tempRow][tempCol][timeInstant + test_index[0] + lookback]
				lr_pred += pred_matrix_LR[timeInstant][tempRow][tempCol]


			predictionList.append(yc)
			actualList.append(xc)
			no_cells_in_cluster = len(clusters[currentClusterNo])

			if area_updated == False:
				area_list.append(no_cells_in_cluster*area_cell)

			perc_crime_cap[yy] += pred_sum

			mse_c += math.pow(yc - xc,2)
			count_mse += 1

			wape_diff_sum_cl += abs(yc-xc)
			wape_actual_sum_cl += xc

			mse_lr += math.pow(xc - lr_pred,2)
			count_mse_lr += 1

			wape_diff_sum_lr += abs(xc - lr_pred)
			wape_actual_sum_lr += xc

		if area_updated == False:
			dict_area[splitNo] = area_list
			area_updated = True	

		matrix_resource_metric.append((splitNo,timeInstant, predictionList, actualList))

		#print rmse_c
		#print n_points
		#raw_input('')
		perc_crime_cap = [float(each*100)/pred_sum for each in perc_crime_cap]	


		#tempList = [0] + perc_area
		#print tempList
		#print perc_crime_cap
		#raw_input('')
			
		#plt.plot([0] + perc_area,[0] + perc_crime_cap)
		#plt.show()
		#raw_input('')	
		
		#Adding a (0,0) point on AUC curve is important for area calculation

		area = integrate.simps([0] + perc_crime_cap,[0] + perc_area)
		AUC += area
		perc_crime_cap_avg = np.add(perc_crime_cap,perc_crime_cap_avg)



	AUC = AUC/len(pred_list_grid_clustering[0])
	#print AUC
	kf_sum_AUC += AUC

	perc_crime_cap_avg = [float(i)/(timeLength) for i in perc_crime_cap_avg]

	fig = plt.figure()
	
	plt.plot([i for i in range(len(clusters))],perc_crime_cap_avg)
	plt.xlabel('Number of clusters')
	plt.ylabel('Percentage crime captured')
	plt.title('AUC for linear regression with grid clustering(' + str(row) + ',' + str(col) + ',' + str(time_period) + ',' + str(lookback) + ',' + str(countThreshold) + ')')
	#fig.savefig('AUC_LR_GridClustering_split_' + str(splitNo) + '.png')
	#plt.show()
	splitNo += 1

print('for following paramters : ' + str(row) + ',' + str(col) + ',' + str(time_period) + ',' + str(lookback))

rmse_c = math.pow(mse_c/float(count_mse),0.5)
print('rmse for LR+GC: ' + str(rmse_c))

rmse_dt = math.pow(mse_dt/float(count_dt),0.5)
print('rmse(entire downtown) for LR:' + str(rmse_dt))


#For hybrid approach, check if grid is relevant one!!!!!!!!!!!!!!!!!!!!
#rmse_lr = math.pow(mse_lr/float(count_mse_lr),0.5)
#print('rmse for LR clustering hybrid : ' + str(rmse_lr)) 

wape_dt = wape_diff_sum_dt/wape_actual_sum_dt
wape_cl = wape_diff_sum_cl/wape_actual_sum_cl
wape_lr = wape_diff_sum_lr/wape_actual_sum_lr

print('WAPE for downtown as one cell: ' + str(wape_dt))
print('WAPE for clustering: ' + str(wape_cl))

#For hybrid approach, check if grid is relevant one!!!!!!!!!!!!!!!!!!!!
#print('WAPE for LR, clustering hybrid approach ' + str(wape_lr))


kf_avg_AUC = kf_sum_AUC/no_splits	

clusterSizeAvg = clusterSizeSum/no_splits
print('Avg number of clusters =' + str(clusterSizeAvg))


#plt.plot(perc_area, perc_crime_cap,'ro')

print(str(kf_avg_AUC) + 'using normal Linear regression and grid clustering')



resourceMatrix_file_name = 'lr_cl_resourceMatrix_' + str(row) + ',' + str(col) + ',' + str(time_period) + ',' + str(lookback) + ',' + str(countThreshold)
f1 = open(resourceMatrix_file_name, 'wb')
pickle.dump(matrix_resource_metric, f1)
f1.close()

areaDict_file_name = 'lr_cl_areaDict_'  + str(row) + ',' + str(col) + ',' + str(time_period) + ',' + str(lookback) + ',' + str(countThreshold)
f2 = open(areaDict_file_name,'wb')
pickle.dump(dict_area, f2)
f2.close()



