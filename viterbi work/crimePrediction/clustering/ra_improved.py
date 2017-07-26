"""
Author : Omkar Damle
Date : 10th July 2017

This code calculates the resource allocation metric 
Algorithm:
1. Select the cell with minimum area.
2. Assign as many resources possible to it subject to requirements
3. Repeat this step until the resources are finished or all the cells are covered.

Input : 2 pickle files

Output : graph of number of crimes captured
"""


import pickle
import sys
import matplotlib.pyplot as plt
import math

row = int(sys.argv[1])
col = int(sys.argv[2])
time_period = int(sys.argv[3])
lookback = int(sys.argv[4])

#matrix_lr_resource_metric = pickle.load(open('lr_resourceMatrix_' + str(row) + ',' + str(col) + ',' + str(time_period) + ',' + str(lookback) , 'rb'))

#dict_area_lr = pickle.load(open('lr_areaDict_'  + str(row) + ',' + str(col) + ',' + str(time_period) + ',' + str(lookback), 'rb'))


#clusterThresholds = [0,100,200,300,400,500,600, 40000]
clusterThresholds = [0, 1000, 5000, 10000 ,40000]
numberOfClusters = [100,22,5,4,1]

matrix_lr_cl_resource_metric = []
dict_area_lr_cl = []

for ii in range(len(clusterThresholds)):
	matrix_lr_cl_resource_metric.append(pickle.load(open('lr_cl_resourceMatrix_' + str(row) + ',' + str(col) + ',' + str(time_period) + ',' + str(lookback)  + ',' + str(clusterThresholds[ii]), 'rb')))
	dict_area_lr_cl.append(pickle.load(open('lr_cl_areaDict_'  + str(row) + ',' + str(col) + ',' + str(time_period) + ',' + str(lookback) + ',' + str(clusterThresholds[ii]), 'rb')))

#total number of units available

k_array = [10*(i+1) for i in range(50) ]	#check

#Area covered by a single unit
area_one_cell = 5.69/(row*col)
m_array = [area_one_cell*0.2]

# 1 unit of police resource can prevent n0 crimes in m0 area
m0 = 1
n0 = 1

#for j in range(len(m_array)):
#	crimes_prevented = []
for method in range(len(clusterThresholds)):
	crimes_prevented = []
	
	matrix = matrix_lr_cl_resource_metric[method]
	dict_area = dict_area_lr_cl[method]
	
	for i in range(len(k_array)):
		accuracy_sum = 0
		count = 0
		
		
		for each in matrix:
			splitNo = each[0]
			time = each[1]
			pred_list= each[2]
			actual_list = each[3]
	
			area_list = dict_area[splitNo]
	
			#calculate the no of units deployed for each cluster/cell
			#No of units deployed is directly proportional to the crime predicted in that cluster/cell

			sum_predictions = 0
			
			for x in range(len(pred_list)):
				if pred_list[x]>0:
					sum_predictions += pred_list[x]

				

			crime_stopped = 0
			crime_count = 0

			units_deployed_list = [0 for jj in range(len(pred_list))]

			units_left = k_array[i]

			fraction_units =[]

			#distribute to the ones with smallest area
			area_cell_tuple = [(area_list[jj],jj) for jj in range(len(pred_list))]

			area_cell_tuple.sort(key = lambda tuple1:tuple1[0])

			#print area_cell_tuple
			#raw_input('')

			for area, cellNo in area_cell_tuple:
				crimeNo = pred_list[cellNo]
				if crimeNo < 0:
					crimeNo = 0

				resource_upper_limit = float(crimeNo)*area/(m0*n0)

				units_deployed = math.floor(resource_upper_limit)

				if units_left >= units_deployed:
					units_left = units_left - units_deployed
					units_deployed_list[cellNo] += units_deployed
					fraction = resource_upper_limit - units_deployed
	
					if fraction > 0:
						fraction_units.append((cellNo,crimeNo - (m0*n0*units_deployed/area)))
				else:
					units_deployed_list[cellNo] = units_left		
					units_left = 0
					break

			fraction_units.sort(reverse = True, key = lambda tuple:tuple[1])		
			#print fraction_units

			#phase 2 : distribute the remaining among the most needful
			for cellNo, measure in fraction_units:
				if units_left==0:
					break

				units_left -= 1

				units_deployed_list[cellNo] += 1

			#print units_left
			#if units_left > 0:
			#	print units_left
			#	raw_input('')

			#print units_deployed_list		
			#raw_input('')
			
			for cellNo in range(len(pred_list)):
				

				#fraction_area_covered = (units_deployed_list[x]*m_array[0])/area_list[x]

				#if fraction_area_covered > 1:
				#	fraction_area_covered = 1.0
				crime_protection = m0*n0*units_deployed_list[cellNo]/area_list[cellNo]

				if crime_protection > actual_list[cellNo]:
					crime_stopped += actual_list[cellNo]
				else:
					crime_stopped += crime_protection
					
				crime_count += actual_list[cellNo]


			accuracy_sum += float(crime_stopped)/crime_count	
			count += 1

		accuracy_avg =  accuracy_sum/len(matrix)
		#print('The average accuracy is : ' + str(accuracy_avg))
		crimes_prevented.append(accuracy_avg)


	#plt.plot(k_array, crimes_prevented,label = 'Area covered by one cell/cluster = ' + str(m_array[j]) + 'sq km')
	#plt.plot(k_array, crimes_prevented,label = 'Count threshold: ' + str(clusterThresholds[method]))
	plt.plot(k_array, crimes_prevented,label = 'Average number of clusters: ' + str(numberOfClusters[method]))

plt.xlabel('Total number of units deployed')
plt.ylabel('Average fraction of crime prevented')
plt.legend()
plt.title('Improved Resource allocation metric_' + str(row) + 'x' + str(col) + '_m0=' + str(m0) + '_n0=' + str(n0) )
plt.show()