"""
Author : Omkar Damle
Date : 31st May 2017
This code uses regression to predict crime frequency
Input : A matrix where each cell corresponds to the grid cell. Each cell contains a list of numbers which represents the 
number of crime incidents in a particular period of time.(may be 1 day or a week or a month,...)

kFold validation has been used



to do:
done - smoothing correctly!!!!!
done - remove set and see if clustering is also improving results
clustering only on training data
change lookback and see

mean calculation is wrong in previous runs
"""

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


#clustering paramter
# As epsilon increases, cluster size increases and number of clusters may decrease

epsilon = float(sys.argv[5])

# As min_samples increases, number of clusters will decrease
min_samples = int(sys.argv[6])

#the method for calculating 'percentage of crime captured' for AUC across time for one fold. 
#1 - sum/sum or 
#2 - actual/total + actual/total + ... and average
method = int(sys.argv[7]) 

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
		#print matrix[i][j]
		#raw_input('')
		matrix_smooth[i][j] = smoothen(timeSeries,smoothing_window)
		
		#remove first and last element of original 
		if smoothing_window!=3:
			raise ValueError('Smoothing window is not 3! Make matrix_smooth and matrix elements of same length')

		matrix_shortened[i][j] = matrix[i][j][1:-1]

		#print len(matrix_smooth[i][j])
		#print len(matrix_shortened[i][j])

		#raw_input('')



print('FYI - Smoothing is done to data.')

"""
smoothing_window = 3

matrix = [[[] for x in range(col)] for y in range(row)]

#smoothen the data
for i in range(row):
	for j in range(col):
		timeSeries = matrix1[i][j]
		matrix[i][j] = smoothen(timeSeries,smoothing_window)
		#print matrix[i][j]
		#raw_input('')
"""



#print matrix[0][0]

#row = len(matrix)
#col = len(matrix[0])



cluster_matrix = np.load('cluster_matrix' + str(row) + 'x' + str(col) + '_DTLA_' + str(epsilon) + '_' + str(min_samples) + '.npy')
cluster_cells = np.load('cluster_cells' + str(row) + 'x' + str(col) + '_DTLA_' + str(epsilon) + '_' + str(min_samples) + '.npy')


#correlation_cell_matrix = np.load('correlation_cell_matrix' + str(row) + 'x' + str(col) + '_' + str(time_period) + '_DTLA.npy')

len_data = len(matrix_smooth[0][0])
#len_training_data = int(math.floor(len_data*((k-1)/k)))
#len_testing_data = len(matrix[0][0]) - len_training_data

#print('number of training matrices : ' + str(len_training_data))
#print('number of testing matrices : ' + str(len_testing_data))
#let us split in 70(training):30(evaluation)

no_splits = 5

kf = KFold(n_splits=no_splits)

n_points = 101 #required for AUC calculation

kf_sum_perc_crime_cap = [0 for i in range(n_points)]
kf_sum_perc_crime_mean_cap = [0 for i in range(n_points)]
kf_sum_perc_crime_neighbor_cap = [0 for i in range(n_points)]
kf_sum_perc_crime_neighbor_clustering_cap = [0 for i in range(n_points)]
kf_sum_perc_crime_correlation_cap = [0 for i in range(n_points)]


#declared here so that they can be averaged over the splits
mean_sq_error = 0
mean_sq_error_1 = 0
mean_sq_error_neighbors = 0
mean_sq_error_neighbors_clustering = 0
mean_sq_error_correlation = 0
norm = 0

mse_count = 0


#print kf.split(np.zeros(len_data,2))


#In order to ensure uniform splitting
#quotient = len_data/no_splits
#len_data = quotient*no_splits

kf_sum_AUC = 0
splitNo = 0

#rmse for grid cells
mse_g = 0
mse_g_count = 0


#WAPE(weighted absolute percentage error) variables
#https://en.wikipedia.org/wiki/Calculating_demand_forecast_accuracy
wape_diff_sum_g = 0	#numerator for WAPE
wape_actual_sum_g = 0 #denominator for WAPE


#This matrix is used to calculate the resource allocation metric
#The columns in this matrix are : split_no,timeInstant, prediction_list, actual_list


matrix_resource_metric = []

dict_area = {}

#units - sq km
area_entire_grid = 5.69
no_cells = row*col
area_cell = 5.69/no_cells
area_list = [ area_cell for i in range(no_cells)]





for train_index,test_index in kf.split([[0 for i in range(2)] for j in range(len_data)]):
	splitNo += 1

	#print('The test indices are:' + str(test_index[0]) + " to " + str(test_index[-1]))
	#print('The train indices are:' + str(train_index))


	#matrix_area.append((splitNo, area_list))

	pred_matrix = [[[ 0 for y in range(col)] for x in range(row)] for time in range(len(test_index) - lookback)]
	pred_matrix_using_mean = [[[ 0 for y in range(col)] for x in range(row)] for time in range(len(test_index) - lookback)]
	pred_matrix_neighbors = [[[ 0 for y in range(col)] for x in range(row)] for time in range(len(test_index) - lookback)]
	pred_matrix_clustering = [[[ 0 for y in range(col)] for x in range(row)] for time in range(len(test_index) - lookback)]
	pred_matrix_correlation_clustering = [[[ 0 for y in range(col)] for x in range(row)] for time in range(len(test_index) - lookback)]

	reg_mat = [[LinearRegression() for y in range(col)] for x in range(row)]
	reg_mat_neighbors = [[LinearRegression() for y in range(col)] for x in range(row)]
	reg_mat_clustering = [[LinearRegression() for y in range(col)] for x in range(row)]
	reg_mat_correlation = [[LinearRegression() for y in range(col)] for x in range(row)]

	max_mean = 0
	errors_sum = 0

	total_crime_across_pred_days = 0

	coefs = [[] for i in range(lookback)]

	n_crimes = [[ 0 for y in range(col)] for x in range(row)]









	"""
	Construct the correlation matrix based on the training data
	"""
	
	correlation_cell_matrix = [[[] for x in range(col)] for y in range(row)]


	#Lets find the cells with highest correlation
	
	for r in range(row):
		for c in range(col):
			
			x1 = matrix_smooth[r][c][train_index]
			#print(len(x1))
			#raw_input('')
			for r1 in range(row):
				for c1 in range(col):
					if r1==r and c1==c:
						continue

					#pair selected, find the correlation

					x2 = matrix_smooth[r1][c1][train_index]
					
					pcc, pVal = sts.pearsonr(x1[1:],x2[:-1])	
					#print x1[1:]
					#print x2[:-1]
					#raw_input('')
					#print pcc,pVal
					#raw_input('')
					if pcc > 0.4:
						correlation_cell_matrix[r][c].append((r1,c1))

#	np.save('correlation_cell_matrix' + str(row) + 'x' + str(col) + '_' +str(time_period) + '_DTLA' , correlated_cell_matrix)

	





	for i in range(row):
		for j in range(col):

			#print('i am in cell : ' + str(i) + ',' + str(j))
			indep_var_mat = []
			indep_var_mat_neighbors = []
			indep_var_mat_clustering = []
			indep_var_mat_correlation = []
			dep_var_col = []

			clusters = list(cluster_matrix[i][j])				#make sure while doing prediction, the order of independent variable is the same
			
			neighborhood_clustered_cells = set()

			for xx in clusters:
				if xx == -1:
					continue
				#print cluster_cells[xx]
				for m in range(len(cluster_cells[xx])):
					neighborhood_clustered_cells.add(cluster_cells[xx][m])

			
			#The cell's own history is being considered separately. Hence remove the cell from neighborhood cells
			#discard() - removes an element if present

			neighborhood_clustered_cells.discard((i,j))

			neighborhood_clustered_cells_list = list(neighborhood_clustered_cells)

			#print neighborhood_clustered_cells	
			#raw_input('')

			timeIter=0
			cellList_smooth = matrix_smooth[i][j]
			cellList_shortened = matrix_shortened[i][j]

			#for including von neuman neighborhood

			prevRow = i-1 if (i-1)>=0 else i
			nextRow = i+1 if (i+1)<row else i
			prevCol = j-1 if (j-1)>=0 else j
			nextCol = j+1 if (j+1)<col else j


			for timeIter in train_index:
				#restrict the independent and dependent variables to training set only
				if timeIter+lookback in test_index:
					continue

				#reached the end	
				if timeIter+lookback == len_data:
					break	

#			while (k+lookback) < len_training_data:
				indep_var_mat.append(cellList_smooth[timeIter:(timeIter + lookback )])

				
				neighbors = [matrix_smooth[prevRow][j][timeIter+lookback-1],matrix_smooth[nextRow][j][timeIter+lookback-1],matrix_smooth[i][prevCol][timeIter+lookback-1], matrix_smooth[i][nextCol][timeIter+lookback-1]]
				clustered_neighbors = []

				#for set_cell in neighborhood_clustered_cells:
				for list_cell in neighborhood_clustered_cells_list:
					#print set_cell
					row1 = list_cell[0]
					col1 = list_cell[1]
					clustered_neighbors.append(matrix_smooth[row1][col1][timeIter+lookback-1])	
					#print matrix[row1][col1][timeIter+lookback-1]

				if len(neighborhood_clustered_cells) != 0:
					pass
					#print matrix[i][j][timeIter+lookback-1]	
					#raw_input('')

				correlated_neighbors = []	
				for each in correlation_cell_matrix[i][j]:
					r = each[0]
					c = each[1]
					correlated_neighbors.append(matrix_smooth[r][c][timeIter+lookback-1])


				indep_var_mat_neighbors.append(np.append(cellList_smooth[timeIter:(timeIter+ lookback)],neighbors))
				indep_var_mat_clustering.append(np.append(cellList_smooth[timeIter:(timeIter+ lookback)],clustered_neighbors))
				indep_var_mat_correlation.append(np.append(cellList_smooth[timeIter:(timeIter+ lookback)],correlated_neighbors))


				dep_var_col.append(cellList_smooth[timeIter+lookback])
				n_crimes[i][j] += cellList_smooth[timeIter+lookback]
				timeIter += 1

				#if i==7 and (j == 18):
				#	print(str(cellList[k:(k+ lookback)]) + ' ' + str(cellList[k+lookback]))
				#	raw_input('')

			reg_mat[i][j].fit(indep_var_mat,dep_var_col)
			reg_mat_neighbors[i][j].fit(indep_var_mat_neighbors,dep_var_col)
			reg_mat_clustering[i][j].fit(indep_var_mat_clustering,dep_var_col)
			reg_mat_correlation[i][j].fit(indep_var_mat_correlation,dep_var_col)


			#print out the coefficients of crime
			#print('No of crimes in this cell = ' + str(n_crimes[i][j]))
			#print(str(i) + ' ' + str(j) + str(reg_mat[i][j].coef_) + ' ' + str(reg_mat[i][j].intercept_))

			#for xx in range(lookback):
			#	coefs[xx].append(reg_mat[i][j].coef_[xx])
			
			#raw_input()

			#let us make the predictions and store them in pred_matrix

			#k=len_training_data
			features_mat = []
			features_neighbors_mat = []
			features_neighbors_clustering_mat = []
			features_mat_correlation = []

			#while (k+lookback) < len_data:
			for timeIter in test_index:

				if (timeIter+lookback) not in test_index:
					break 

				features_mat.append(cellList_shortened[timeIter:(timeIter+lookback)])

				neighbors = [matrix_shortened[prevRow][j][timeIter+lookback-1],matrix_shortened[nextRow][j][timeIter+lookback-1],matrix_shortened[i][prevCol][timeIter+lookback-1], matrix_shortened[i][nextCol][timeIter+lookback-1]]

				features_neighbors_mat.append(np.append(cellList_shortened[timeIter:(timeIter+lookback)],neighbors))

				clustered_neighbors = []
				
				
				#for set_cell in neighborhood_clustered_cells:
				for list_cell in neighborhood_clustered_cells_list:
					#print set_cell
					row1 = list_cell[0]
					col1 = list_cell[1]
					clustered_neighbors.append(matrix_shortened[row1][col1][timeIter+lookback-1])	


				features_neighbors_clustering_mat.append(np.append(cellList_shortened[timeIter:(timeIter+lookback)],clustered_neighbors))
				

				correlated_neighbors = []
				for each in correlation_cell_matrix[i][j]:
					r = each[0]
					c = each[1]
					correlated_neighbors.append(matrix_shortened[r][c][timeIter+lookback-1])

				features_mat_correlation.append(np.append(cellList_shortened[timeIter:(timeIter+lookback)],correlated_neighbors))

				total_crime_across_pred_days += cellList_shortened[timeIter+lookback]
				timeIter +=1

			pred_list = reg_mat[i][j].predict(features_mat)
			pred_neighbors_list = reg_mat_neighbors[i][j].predict(features_neighbors_mat)
			pred_neighbors_clustering_list = reg_mat_clustering[i][j].predict(features_neighbors_clustering_mat)
			pred_correlation_list = reg_mat_correlation[i][j].predict(features_mat_correlation)

			for m in range(len(pred_list)):
				#print(m + len_training_data + lookback)
				#pred_matrix[m + len_training_data + lookback][i][j] = pred_list[m]
				pred_matrix[m][i][j] = pred_list[m]
				pred_matrix_using_mean[m][i][j] = np.mean(cellList_shortened[m + test_index[0] : m + test_index[0] + lookback])
				pred_matrix_neighbors[m][i][j] = pred_neighbors_list[m]
				pred_matrix_clustering[m][i][j] = pred_neighbors_clustering_list[m]
				pred_matrix_correlation_clustering[m][i][j] = pred_correlation_list[m]



	np.save('pred_matrix_split_' + str(splitNo), pred_matrix)
				
	"""
	let us calculate the AUC-ROC metric
	"""

	row = len(pred_matrix[0])
	col = len(pred_matrix[0][0])

	#raw_input('waiting for your permission')

	


	if method == 1:	

		n_points = 101

		perc_crime_cap = [0 for i in range(n_points)]
		perc_crime_mean_cap = [0 for i in range(n_points)]
		perc_crime_neighbor_cap = [0 for i in range(n_points)]
		perc_crime_neighbor_clustering_cap = [0 for i in range(n_points)]
		perc_crime_correlation_cap = [0 for i in range(n_points)]

		perc_area = [i for i in range(n_points)]

		for i in range(n_points):
			pred_sum =  0	
			pred_mean_sum = 0
			pred_neighbor_sum = 0
			pred_neighbor_clustering_sum = 0
			pred_correlation_sum = 0
			no_top_cells = int(math.floor(perc_area[i]*row*col/100))


			for x in range(len(pred_matrix)):
				
				temp_list = []
				temp_list_mean = []
				temp_list_neighbor = []
				temp_list_neighbor_clustering = []
				temp_list_correlation = []
				for p in range(row):
					for q in range(col):
						temp_list.append((pred_matrix[x][p][q],p,q))
						temp_list_mean.append((pred_matrix_using_mean[x][p][q],p,q))
						temp_list_neighbor.append((pred_matrix_neighbors[x][p][q],p,q))
						temp_list_neighbor_clustering.append((pred_matrix_clustering[x][p][q],p,q))
						temp_list_correlation.append((pred_matrix_correlation_clustering[x][p][q],p,q))


				temp_list.sort(reverse = True, key=lambda tuple:tuple[0])
				temp_list_mean.sort(reverse = True, key = lambda tuple:tuple[0])
				temp_list_neighbor.sort(reverse = True, key = lambda tuple:tuple[0])
				temp_list_neighbor_clustering.sort(reverse = True, key = lambda tuple:tuple[0])
				temp_list_correlation.sort(reverse=True, key = lambda tuple:tuple[0])

				#print temp_list

				#raw_input()

				for y in range(no_top_cells):
					tempRow = temp_list[y][1]
					tempCol = temp_list[y][2]
					pred_sum += matrix_shortened[tempRow][tempCol][x + test_index[0] + lookback]


					tempRow_mean = temp_list_mean[y][1]
					tempCol_mean = temp_list_mean[y][2]
					pred_mean_sum += matrix_shortened[tempRow_mean][tempCol_mean][x + test_index[0] + lookback]

					tempRow_n = temp_list_neighbor[y][1]
					tempCol_n = temp_list_neighbor[y][2]
					pred_neighbor_sum += matrix_shortened[tempRow_n][tempCol_n][x + test_index[0] + lookback]

					tempRow_cluster = temp_list_neighbor_clustering[y][1]
					tempCol_cluster = temp_list_neighbor_clustering[y][2]
					pred_neighbor_clustering_sum += matrix_shortened[tempRow_cluster][tempCol_cluster][x + test_index[0] + lookback]

					tempRow_corr = temp_list_correlation[y][1]
					tempCol_corr = temp_list_correlation[y][2]
					pred_correlation_sum += matrix_shortened[tempRow_corr][tempCol_corr][x + test_index[0] + lookback]




			perc_crime_cap[i] = pred_sum*100/total_crime_across_pred_days
			perc_crime_mean_cap[i] = pred_mean_sum*100/total_crime_across_pred_days 
			perc_crime_neighbor_cap[i] = pred_neighbor_sum*100/total_crime_across_pred_days
			perc_crime_neighbor_clustering_cap[i] = pred_neighbor_clustering_sum*100/total_crime_across_pred_days
			perc_crime_correlation_cap[i] = pred_correlation_sum*100/total_crime_across_pred_days

		norm = 0
		temp = 0
		temp1= 0



		for x in range(len(pred_matrix)):
			
			#list for all cells at a particular time instant
			predictionList = []
			actualList = []

			for p in range(row):
				for q in range(col):

					predictionList.append(pred_matrix[x][p][q])
					actualList.append(matrix_shortened[p][q][x + test_index[0] + lookback])

					mean_sq_error += math.pow((pred_matrix[x][p][q] - matrix_shortened[p][q][x + test_index[0] + lookback]),2)
					mean_sq_error_1 += math.pow((pred_matrix_using_mean[x][p][q] - matrix_shortened[p][q][x + test_index[0] + lookback]),2)
					mean_sq_error_neighbors += math.pow((pred_matrix_neighbors[x][p][q] - matrix_shortened[p][q][x + test_index[0] + lookback]),2)
					mean_sq_error_neighbors_clustering += math.pow((pred_matrix_clustering[x][p][q] - matrix_shortened[p][q][x + test_index[0] + lookback]),2)
					mean_sq_error_correlation += math.pow((pred_matrix_correlation_clustering[x][p][q] - matrix_shortened[p][q][x + test_index[0] + lookback]),2)			
					temp += math.pow((pred_matrix_correlation_clustering[x][p][q] - matrix_shortened[p][q][x + test_index[0] + lookback]),2)
					temp1 += math.pow((pred_matrix[x][p][q] - matrix_shortened[p][q][x + test_index[0] + lookback]),2)
					norm+=1
					mse_count += 1

					mse_g += math.pow((pred_matrix[x][p][q] - matrix_shortened[p][q][x + test_index[0] + lookback]),2)
					mse_g_count += 1

					wape_diff_sum_g += abs(pred_matrix[x][p][q] - matrix_shortened[p][q][x + test_index[0] + lookback])
					wape_actual_sum_g += matrix_shortened[p][q][x + test_index[0] + lookback]

			matrix_resource_metric.append((splitNo, x, predictionList, actualList))
			

		print temp
		print temp1

		for p in range(row):
			for q in range(col):
				error_list = []
				for x in range(len(pred_matrix)):
					error_list.append(pred_matrix[x][p][q] - matrix_shortened[p][q][x + test_index[0] + lookback])
				#plt.hist(error_list, bins = 100, range=(-1,1))
				#plt.show()
				#raw_input('')
		"""
		mean_sq_error = mean_sq_error/norm
		mean_sq_error_1 = mean_sq_error_1/norm
		mean_sq_error_neighbors = mean_sq_error_neighbors/norm
		mean_sq_error_neighbors_clustering = mean_sq_error_neighbors_clustering /norm
		mean_sq_error_correlation = mean_sq_error_correlation/norm
		"""
		#print perc_area
		#print perc_crime_cap

		kf_sum_perc_crime_cap = np.add(kf_sum_perc_crime_cap,perc_crime_cap)
		kf_sum_perc_crime_mean_cap = np.add(kf_sum_perc_crime_mean_cap,perc_crime_mean_cap)
		kf_sum_perc_crime_neighbor_cap = np.add(kf_sum_perc_crime_neighbor_cap,perc_crime_neighbor_cap)
		kf_sum_perc_crime_neighbor_clustering_cap = np.add(kf_sum_perc_crime_neighbor_clustering_cap,perc_crime_neighbor_clustering_cap)
		kf_sum_perc_crime_correlation_cap = np.add(kf_sum_perc_crime_correlation_cap,perc_crime_correlation_cap)


	elif method == 2:
		
		#raw_input('waiting for your permission')

		n_points = row*col

		AUC = 0

		#in order to average across time instants
		perc_crime_cap_avg = [0 for i in range(n_points)]

		timeLength = len(pred_matrix)


		for timeInstant in range(timeLength):

			perc_crime_cap = [0 for i in range(n_points)]

			perc_area = [-1 for i in range(n_points)]

			densities = []

			#for clusterNo in range(n_points):
			for ii in range(row):		
				for jj in range(col):
				#density = float(pred_list_grid_clustering[i][timeInstant])/len(clusters[i])
					density = pred_matrix[timeInstant][ii][jj]
					densities.append((density,ii,jj))

			densities.sort(reverse=True, key=lambda tuple1:tuple1[0])	

			total_crime = 0
			pred_sum = 0

			for yy in range(n_points):
					#currentClusterNo = densities[yy][1]
					
				currentRow = densities[yy][1]
				currentCol = densities[yy][2]
				perc_area[yy] = float(yy+1)/n_points


				pred_sum += matrix_shortened[currentRow][currentCol][timeInstant + test_index[0] + lookback]
				total_crime += pred_sum

				perc_crime_cap[yy] += pred_sum


			perc_crime_cap = [float(each*100)/pred_sum for each in perc_crime_cap]	


			#print perc_area
			#print perc_crime_cap
			#raw_input('')
				
			#plt.plot(perc_area,perc_crime_cap)
			#plt.show()
			#raw_input('')	
			area = integrate.simps(perc_crime_cap,perc_area)
			AUC += area
			perc_crime_cap_avg = np.add(perc_crime_cap,perc_crime_cap_avg)

		AUC = AUC/timeLength
		#print AUC
		kf_sum_AUC += AUC

		perc_crime_cap_avg = [float(i)/(timeLength) for i in perc_crime_cap_avg]

		fig = plt.figure()
		plt.plot([i for i in range(n_points)],perc_crime_cap_avg)
		plt.xlabel('Number of clusters')
		plt.ylabel('Percentage crime captured')
		plt.title('AUC for linear regression(' + str(row) + ',' + str(col) + ',' + str(time_period) + ',' + str(lookback) + ')')
		fig.savefig('AUC_LR_split_' + str(splitNo) + '.png')
		plt.show()


if method == 1:

	#np.save('lr_resourceMatrix_' + str(row) + ',' + str(col) + ',' + str(time_period) + ',' + str(lookback) , matrix_resource_metric)
	#np.save('lr_areaMatrix_'  + str(row) + ',' + str(col) + ',' + str(time_period) + ',' + str(lookback) , matrix_area)

	resourceMatrix_file_name = 'lr_resourceMatrix_' + str(row) + ',' + str(col) + ',' + str(time_period) + ',' + str(lookback)
	f1 = open(resourceMatrix_file_name, 'wb')
	pickle.dump(matrix_resource_metric, f1)
	f1.close()


	for ii in range(splitNo):
		splitNo = ii+1
		dict_area[splitNo] = area_list

	areaDict_file_name = 'lr_areaDict_'  + str(row) + ',' + str(col) + ',' + str(time_period) + ',' + str(lookback)
	f2 = open(areaDict_file_name,'wb')
	pickle.dump(dict_area, f2)
	f2.close()



	mean_sq_error = mean_sq_error/mse_count
	mean_sq_error_1 = mean_sq_error_1/mse_count
	mean_sq_error_neighbors = mean_sq_error_neighbors/mse_count
	#mean_sq_error_neighbors_clustering = mean_sq_error_neighbors_clustering /(no_splits * norm)
	mean_sq_error_neighbors_clustering = mean_sq_error_neighbors_clustering /mse_count

	mean_sq_error_correlation = mean_sq_error_correlation/mse_count


	kf_avg_perc_crime_cap = [xx/no_splits for xx in kf_sum_perc_crime_cap]
	kf_avg_perc_crime_mean_cap = [xx/no_splits for xx in kf_sum_perc_crime_mean_cap]
	kf_avg_perc_crime_neighbor_cap = [xx/no_splits for xx in kf_sum_perc_crime_neighbor_cap]
	kf_avg_perc_crime_neighbor_clustering_cap = [xx/no_splits for xx in kf_sum_perc_crime_neighbor_clustering_cap]
	kf_avg_perc_crime_correlation_cap = [xx/no_splits for xx in kf_sum_perc_crime_correlation_cap]

	print('for following paramters : ' + str(row) + ',' + str(col) + ',' + str(time_period) + ',' + str(lookback) + ',' + str(epsilon) + ',' + str(min_samples))

	#plt.plot(perc_area, kf_avg_perc_crime_cap,'ro',label = 'lr')
	#plt.show()

	#print kf_avg_perc_crime_cap
	#print perc_area
	raw_input('')

	area = integrate.simps(kf_avg_perc_crime_cap,perc_area)/(n_points*n_points)
	print(str(area) + 'using normal Linear regression')

	area = integrate.simps(kf_avg_perc_crime_mean_cap,perc_area)/(n_points*n_points)
	print(str(area) + 'using mean method')

	area = integrate.simps(kf_avg_perc_crime_neighbor_cap,perc_area)/(n_points*n_points)
	print(str(area) + 'using LR with neighbors method')

	area = integrate.simps(kf_avg_perc_crime_neighbor_clustering_cap,perc_area)/(n_points*n_points)
	print(str(area) + 'using LR with neighbors CLUSTERING method')

	#plt.figure()
	#plt.plot(perc_area, kf_avg_perc_crime_neighbor_clustering_cap,'bo',label='clustering')
	#plt.legend()
	#plt.show()

	area = integrate.simps(kf_avg_perc_crime_correlation_cap,perc_area)/(n_points*n_points)
	print(str(area) + 'using LR with correlation method')


	print('MSE for LR : ' + str(mean_sq_error))
	print('MSE for mean method : ' + str(mean_sq_error_1))
	print('MSE for neighbor method : ' + str(mean_sq_error_neighbors))
	print('MSE for clustering neighbor method : ' + str(mean_sq_error_neighbors_clustering))
	print('MSE for correlation method : ' + str(mean_sq_error_correlation))
	print('')

	rmse_g = math.pow(mse_g/float(mse_g_count),0.5)

	print('Average rmse for LR: ' + str(rmse_g))


	wape_g = wape_diff_sum_g/wape_actual_sum_g
	print('WAPE for LR alone: ' + str(wape_g))


	#plt.title('AUC(0.7605) plot for LR with 10*10 grid, period = 1 day, lookback = 7 days')
	#plt.xlabel('Percentage area surveilled')
	#plt.ylabel('Percentage crime incidents captured')
	#fig.savefig('AUC_' + str(time_period) + 'h_' + str(row) + 'x' + str(col) + '.png')
	#plt.show()

	"""
	with open('2016-PART_I_AND_II_CRIMES.csv') as csvfile:
		readCSV = csv.reader(csvfile,delimiter=',')
		count =0
		#print(shape(readCSV))
		for row in readCSV:
			#print(row[0])
			count+=1
			
		print count	
	"""
	#02/16/2016 07:59:51 PM

	"""
	df = pd.read_csv('2016-PART_I_AND_II_CRIMES.csv')
	timeCol = pd.to_datetime(df['INCIDENT_DATE'],format='%m/%d/%Y %I:%M:%S %p')

	print(type(timeCol[0]))

	timeCol.groupby(timeCol.dt.date).count().plot(kind='line')

	plt.show()
	"""
elif method == 2 :
	kf_avg_AUC = kf_sum_AUC/no_splits	
	print('for following paramters : ' + str(row) + ',' + str(col) + ',' + str(time_period) + ',' + str(lookback))
	print(str(kf_avg_AUC) + 'using normal Linear regression')


