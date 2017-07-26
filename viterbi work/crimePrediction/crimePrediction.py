"""
Author : Omkar Damle
Date : 31st May 2017
This code uses regression to predict crime frequency
Input : A matrix where each cell corresponds to the grid cell. Each cell contains a list of numbers which represents the 
number of crime incidents in a particular period of time.(may be 1 day or a week or a month,...)

"""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import sys

from scipy import integrate
#import pandas as pd

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")


"""
data = [[2,3,4],[4,6,8],[6,9,12]]
score = [9,18,27]

lr.fit(data,score)

print lr.coef_
"""

row = int(sys.argv[1])
col = int(sys.argv[2])
time_period = int(sys.argv[3])
lookback = int(sys.argv[4])

matrix = np.load('matrix_' + str(time_period) + 'h_' + str(row) + 'x' + str(col)+ '.npy')
#print matrix[0][0]

#row = len(matrix)
#col = len(matrix[0])

len_data = len(matrix[0][0])
len_training_data = int(math.floor(len_data*0.7))
len_testing_data = len(matrix[0][0]) - len_training_data

print('number of training matrices : ' + str(len_training_data))
print('number of testing matrices : ' + str(len_testing_data))
#let us split in 70(training):30(evaluation)

pred_matrix = [[[ 0 for y in range(col)] for x in range(row)] for time in range(len_testing_data - lookback)]
pred_matrix_using_mean = [[[ 0 for y in range(col)] for x in range(row)] for time in range(len_testing_data - lookback)]
pred_matrix_neighbors = [[[ 0 for y in range(col)] for x in range(row)] for time in range(len_testing_data - lookback)]

reg_mat = [[LinearRegression() for y in range(col)] for x in range(row)]
reg_mat_neighbors = [[LinearRegression() for y in range(col)] for x in range(row)]

max_mean = 0
errors_sum = 0

total_crime_across_pred_days = 0

coefs = [[] for i in range(lookback)]

n_crimes = [[ 0 for y in range(col)] for x in range(row)]

for i in range(row):
	for j in range(col):

		#print('i am in cell : ' + str(i) + ',' + str(j))
		dep_var_mat = []
		dep_var_mat_neighbors = []
		indep_var_col = []

		k=0
		cellList = matrix[i][j]
		while (k+lookback) < len_training_data:
			dep_var_mat.append(cellList[k:(k+ lookback )])

			#for including von neuman neighborhood
			prevRow = i-1 if (i-1)>=0 else i
			nextRow = i+1 if (i+1)<row else i
			prevCol = j-1 if (j-1)>=0 else j
			nextCol = j+1 if (j+1)<col else j


			
			neighbors = [matrix[prevRow][j][k+lookback-1],matrix[nextRow][j][k+lookback-1],matrix[i][prevCol][k+lookback-1], matrix[i][nextCol][k+lookback-1]]
			
			dep_var_mat_neighbors.append(np.append(cellList[k:(k+ lookback)],neighbors))

			indep_var_col.append(cellList[k+lookback])
			n_crimes[i][j] += cellList[k+lookback]
			k += 1

			#if i==7 and (j == 18):
			#	print(str(cellList[k:(k+ lookback)]) + ' ' + str(cellList[k+lookback]))
			#	raw_input('')

		reg_mat[i][j].fit(dep_var_mat,indep_var_col)
		reg_mat_neighbors[i][j].fit(dep_var_mat_neighbors,indep_var_col)

		#if i==7 and (j == 18):
		#	plt.hist(indep_var_col)
		#	plt.show()

		#plt.plot(indep_var_col)
		#plt.show()
		#raw_input('')

		#print out the coefficients of crime
		#print('No of crimes in this cell = ' + str(n_crimes[i][j]))
		#print(str(i) + ' ' + str(j) + str(reg_mat[i][j].coef_) + ' ' + str(reg_mat[i][j].intercept_))

		#for xx in range(lookback):
		#	coefs[xx].append(reg_mat[i][j].coef_[xx])
		
		#raw_input()

		#let us make the predictions and store them in pred_matrix

		k=len_training_data
		features_mat = []
		features_neighbors_mat = []

		while (k+lookback) < len_data:
			features_mat.append(cellList[k:(k+lookback)])

			neighbors = [matrix[prevRow][j][k+lookback-1],matrix[nextRow][j][k+lookback-1],matrix[i][prevCol][k+lookback-1], matrix[i][nextCol][k+lookback-1]]

			features_neighbors_mat.append(np.append(cellList[k:(k+lookback)],neighbors))
			total_crime_across_pred_days += cellList[k+lookback]
			k +=1

		pred_list = reg_mat[i][j].predict(features_mat)
		pred_neighbors_list = reg_mat_neighbors[i][j].predict(features_neighbors_mat)

		#print pred_list
		#raw_input()

		##print(len(pred_list))
		#print(len(pred_matrix))
		#print(i,j)
		for m in range(len(pred_list)):
			#print(m + len_training_data + lookback)
			#pred_matrix[m + len_training_data + lookback][i][j] = pred_list[m]
			pred_matrix[m][i][j] = pred_list[m]
			pred_matrix_using_mean[m][i][j] = np.mean(cellList[m + len_training_data : m + len_training_data + lookback -1])
			pred_matrix_neighbors[m][i][j] = pred_neighbors_list[m]

		"""
		#let us find the accuracy of our training
		
		score = cross_val_score(LinearRegression(),dep_var_mat,indep_var_col,cv = 5,scoring='neg_mean_squared_error')
		errors_sum += score.mean()

		#print score.mean()
		if max_mean < score.mean():
			max_mean = score.mean()
		"""	
"""
error_avg = errors_sum/(row*col)

print error_avg
print max_mean
"""

"""
let us calculate the AUC-ROC metric
"""

#print('Value of coefficients : ')

#for xx in range(lookback):
#	print np.median(coefs[xx])

#print('len(pred_matrix)=' + str(len(pred_matrix)))
#print('len(pred_matrix[0]) = ' + str(len(pred_matrix[0])))
#print( 'len(pred_matrix[0][0])) = ' + str(len(pred_matrix[0][0])))


#print('total_crime_across_pred_days' +  str(total_crime_across_pred_days))

row = len(pred_matrix[0])
col = len(pred_matrix[0][0])

#raw_input('waiting for your permission')

n_points = 101

perc_crime_cap = [0 for i in range(n_points)]
perc_crime_mean_cap = [0 for i in range(n_points)]
perc_crime_neighbor_cap = [0 for i in range(n_points)]

perc_area = [i for i in range(n_points)]


for i in range(n_points):
	pred_sum =  0	
	pred_mean_sum = 0
	pred_neighbor_sum = 0
	no_top_cells = int(math.floor(perc_area[i]*row*col/100))
	for x in range(len(pred_matrix)):
		
		temp_list = []
		temp_list_mean = []
		temp_list_neighbor = []
		for p in range(row):
			for q in range(col):
				temp_list.append((pred_matrix[x][p][q],p,q))
				temp_list_mean.append((pred_matrix_using_mean[x][p][q],p,q))
				temp_list_neighbor.append((pred_matrix_neighbors[x][p][q],p,q))



		temp_list.sort(reverse = True, key=lambda tuple:tuple[0])
		temp_list_mean.sort(reverse = True, key = lambda tuple:tuple[0])
		temp_list_neighbor.sort(reverse = True, key = lambda tuple:tuple[0])
		#print temp_list
		#raw_input()

		for y in range(no_top_cells):
			tempRow = temp_list[y][1]
			tempCol = temp_list[y][2]
			pred_sum += matrix[tempRow][tempCol][x + len_training_data + lookback]


			tempRow_mean = temp_list_mean[y][1]
			tempCol_mean = temp_list_mean[y][2]
			pred_mean_sum += matrix[tempRow_mean][tempCol_mean][x + len_training_data + lookback]

			tempRow_n = temp_list_neighbor[y][1]
			tempCol_n = temp_list_neighbor[y][2]
			pred_neighbor_sum += matrix[tempRow_n][tempCol_n][x + len_training_data + lookback]


	perc_crime_cap[i] = pred_sum*100/total_crime_across_pred_days
	perc_crime_mean_cap[i] = pred_mean_sum*100/total_crime_across_pred_days 
	perc_crime_neighbor_cap[i] = pred_neighbor_sum*100/total_crime_across_pred_days

mean_sq_error = 0
norm = 0
mean_sq_error_1 = 0


for x in range(len(pred_matrix)):
	for p in range(row):
		for q in range(col):
			mean_sq_error += math.pow((pred_matrix[x][p][q] - matrix[p][q][x + len_training_data + lookback]),2)
			mean_sq_error_1 += math.pow((pred_matrix_using_mean[x][p][q] - matrix[p][q][x + len_training_data + lookback]),2)
			norm+=1

for p in range(row):
	for q in range(col):
		error_list = []
		for x in range(len(pred_matrix)):
			error_list.append(pred_matrix[x][p][q] - matrix[p][q][x + len_training_data + lookback])
		plt.hist(error_list, bins = 100, range=(-1,1))
		plt.show()
		raw_input('')

mean_sq_error = mean_sq_error/norm
mean_sq_error_1 = mean_sq_error_1/norm

#print perc_area
#print perc_crime_cap

fig = plt.figure()

plt.plot(perc_area, perc_crime_cap,'ro')
area = integrate.simps(perc_crime_cap,perc_area)/(n_points*n_points)
print(str(area) + 'for following paramters : ' + str(row) + ',' + str(col) + ',' + str(time_period) + ',' + str(lookback))
area = integrate.simps(perc_crime_mean_cap,perc_area)/(n_points*n_points)
print(str(area) + 'using mean method for following paramters : ' + str(row) + ',' + str(col) + ',' + str(time_period) + ',' + str(lookback))
area = integrate.simps(perc_crime_neighbor_cap,perc_area)/(n_points*n_points)
print(str(area) + 'using LR with neighbors method for following paramters : ' + str(row) + ',' + str(col) + ',' + str(time_period) + ',' + str(lookback))


print('MSE for LR : ' + str(mean_sq_error))
print('MSE for mean method : ' + str(mean_sq_error_1))

print('')

#plt.title('AUC(0.7605) plot for LR with 10*10 grid, period = 1 day, lookback = 7 days')
plt.xlabel('Percentage area surveilled')
plt.ylabel('Percentage crime incidents captured')
fig.savefig('AUC_' + str(time_period) + 'h_' + str(row) + 'x' + str(col) + '.png')
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

