"""
Author : Omkar Damle
Date : 10th July 2017

This code calculates the resource allocation metric 

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

			#phase 1 : distribute the floor of the real number
			for x in range(len(pred_list)):
				units_deployed  = (pred_list[x]*float(k_array[i]))/sum_predictions

				if units_deployed < 0:
					units_deployed = 0
				
				units_deployed_list[x] = math.floor(units_deployed)
				units_left = units_left - math.floor(units_deployed)

				fraction = units_deployed - math.floor(units_deployed)
				if fraction > 0:
					fraction_units.append((x,fraction*pred_list[x]/area_list[x]))

			fraction_units.sort(reverse = True, key = lambda tuple:tuple[1])		
			#print fraction_units

			#phase 2 : distribute the remaining among the most needful
			for t in fraction_units:
				if units_left==0:
					break

				units_left -= 1

				units_deployed_list[t[0]] += 1

			#print units_left
			#print units_deployed_list		
			#raw_input('')
			
			for x in range(len(pred_list)):
				
				fraction_area_covered = (units_deployed_list[x]*m_array[0])/area_list[x]

				if fraction_area_covered > 1:
					fraction_area_covered = 1.0
				
				crime_stopped += float(fraction_area_covered)*actual_list[x]
				crime_count += actual_list[x]

			accuracy_sum += float(crime_stopped)/crime_count	
			count += 1

		accuracy_avg =  accuracy_sum/len(matrix)
		#print('The average accuracy is : ' + str(accuracy_avg))
		crimes_prevented.append(accuracy_avg)


	#plt.plot(k_array, crimes_prevented,label = 'Area covered by one cell/cluster = ' + str(m_array[j]) + 'sq km')
	plt.plot(k_array, crimes_prevented,label = 'Count threshold: ' + str(clusterThresholds[method]))

plt.xlabel('Total number of units deployed')
plt.ylabel('Average fraction of crime prevented')
plt.legend()
plt.title('Resource allocation metric_' + str(row) + 'x' + str(col) + '_m=' + str(m_array[0]/area_one_cell) )
plt.show()