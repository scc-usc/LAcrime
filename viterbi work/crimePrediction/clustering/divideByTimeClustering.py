"""
Load 2016 crime data of Downtown LA and divide into time periods and visualize
Author : Omkar Damle
Date : 5th June 2017
"""

from datetime import date,time,datetime,timedelta
import numpy as np
import math
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import sys
import os.path


row = int(sys.argv[1])
col = int(sys.argv[2])
time_period = int(sys.argv[3])

#if os.path.isfile('matrix_' + str(time_period) + 'h_' + str(row) + 'x' + str(col) + '.npy'):
#	sys.exit('divideByTime.py not needed as division has already been saved')


matrix = np.load('../matrix_' + str(row) + 'x' + str(col) + '_DTLA14to16' + '.npy')
#numpy.save('matrix_' + str(rows) + 'x' + str(cols) + '_DTLA16',matrix)

#print matrix

#row = len(matrix)
#col = len(matrix[0])
#print row,col

#time period in hours

#year,month ,date
start_day = date(2014,01,01)
start_time = time(00,00)

end_day = date(2016,12,31)
end_time = time(23,59)

current = datetime.combine(start_day,start_time)
end = datetime.combine(end_day,end_time)
delta = timedelta(hours = time_period)


day_matrix = [[[] for x in range(col)] for y in range(row)]
count_matrix = [[ 0 for x in range(col)] for y in range(row)]

current = datetime.combine(start_day,start_time)
list_dates = []
while current < end:
	list_dates.append(current.date())
	current += delta

list_freq = [0 for i in range(len(list_dates))]

dic = {'date' : list_dates, 'freq': list_freq}
df = DataFrame(dic)
#print df.info()
#print df.head()

df.set_index('date',inplace = True)

#raw_input()
total = 0
for i in range(row):
	for j in range(col):
		myList = matrix[i][j]
		#print myList

		list_iter = 0
		current = datetime.combine(start_day,start_time)

		while current < end:  
			no_crimes = 0
			day_end = current + delta
			
			while list_iter < len(myList) and myList[list_iter] < day_end :
				no_crimes += 1
				list_iter+=1	
			temp = df.get_value(current.date(),'freq') + no_crimes	
			df.set_value(current.date(),'freq',temp) 
			day_matrix[i][j].append(no_crimes)

			current += delta
		count_matrix[i][j] = len(myList)
		total += len(myList)
#print count_matrix

#print(total)

#print day_matrix[2][2]
"""
for i in range(row):
	for j in range(col):
		print str(int(math.floor(count_matrix[i][j]))),

	print('')

df.plot()
plt.show()
"""

#plt.plot(day_matrix[0][7])
#plt.show()
np.save( 'matrix_' + str(time_period) + 'h_' + str(row) + 'x' + str(col) + '_DTLA_14to16', day_matrix)