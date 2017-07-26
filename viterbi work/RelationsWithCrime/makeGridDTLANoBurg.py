"""
make grid for all crimes excluding burglary
"""

"""
Author : Omkar Damle
Date : 5th June 2017
"""

import pandas as pd
import math
from pyproj import Proj, transform
import datetime
import numpy
import sys
import os.path
import matplotlib.pyplot as plt

def isInsideBox(ll_lon,ll_lat,ur_lon,ur_lat,lon,lat):
	isInside = True
	
	if lat > ur_lat or lat < ll_lat or lon > ur_lon or lon < ll_lon:
		isInside = False

	return isInside


"""
crimeTypes = 
{510:'vehicle_stolen',  
624:'battery - simple assault',
330:'burglary_from_vehicle',
310:'burglary',
440:'THEFT PLAIN - PETTY ($950 & UNDER)',
740:"VANDALISM - FELONY ($400 & OVER, ALL CHURCH VANDALISMS)"
626:'SPOUSAL(COHAB) ABUSE - SIMPLE ASSAULT'
354:'THEFT OF IDENTITY',
230:"ASSAULT WITH DEADLY WEAPON, AGGRAVATED ASSAULT",
420:'THEFT FROM MOTOR VEHICLE - PETTY ($950 & UNDER)'
210:'ROBBERY',
745:'VANDALISM - MISDEAMEANOR ($399 OR UNDER)',
442:'SHOPLIFTING - PETTY THEFT ($950 & UNDER)',
341:"THEFT-GRAND ($950.01 & OVER)EXCPT,GUNS,FOWL,LIVESTK,PROD",
930:'CRIMINAL THREATS - NO WEAPON DISPLAYED',
331:'THEFT FROM MOTOR VEHICLE - GRAND ($400 AND OVER)',
888:'TRESPASSING',
236:'SPOUSAL (COHAB) ABUSE - AGGRAVATED ASSAULT',
649:'DOCUMENT FORGERY / STOLEN FELONY'
480:'BIKE - STOLEN'
901:'VIOLATION OF RESTRAINING ORDER',
761:'BRANDISH WEAPON',
350:"THEFT, PERSON",
320:"BURGLARY, ATTEMPTED",
220:'ATTEMPTED ROBBERY',
860:'BATTERY WITH SEXUAL CONTACT',
121:"RAPE, FORCIBLE",
627:'CHILD ABUSE (PHYSICAL) - SIMPLE ASSAULT'
668:"EMBEZZLEMENT, GRAND THEFT ($950.01 & OVER)"
343:'SHOPLIFTING-GRAND THEFT ($950.01 & OVER)''

"""
#crimeTypes = {'Property crime' : [510,330,310,440,740,354,420,210,745,442,341,331,888,649,480,350,320,668,343],'Violent crime':[624,626,230,210,930,236,761,220,860,121,627]}
lightRelatedCrimes = [510, 624, 330, 440, 740, 354, 230, 420, 210, 745, 442, 341, 
331, 888, 480, 761, 350, 220, 860, 121, 343]

LA_COORDINATES = (34.05, -118.24)
#crimedata = pd.read_csv('lapd_Crime_Data_From_2010_to_Present.csv')
#08/30/2010,2300
#crimedata['Date Occurred'] = pd.to_datetime(crimedata['Date Occurred'],format='%m/%d/%Y')
#crimedata.sort_values(by='Date Occurred',ascending = True, inplace = True)
# for speed purposes
#MAX_RECORDS = 10

#print crimedata.info()

#2016 data
#data16 = crimedata[1208751:-1]
#data16.to_pickle('data16.pkl')

#data16 = pd.read_pickle('data16.pkl')
#print data16.head(50)
#print data16.info()

#data16['dateAndTime'] = data16['Date Occurred'] + datetime.timedelta(hours = numpy.floor(data16['Time Occurred']/100), minutes = data16['Time Occurred']%100)

#data16['dateAndTime'] = pd.to_datetime( data16['Date Occurred'].strftime("%m/%d/%Y") + data16['Time Occurred'],format = '%m/%d/%Y')


#for row in data16.iterrows():
#	row[0]['Date Occurred'] = row[0]['Date Occurred'] + datetime.timedelta(hours = math.floor(row[0]['Time Occurred']/100), minutes = row[0]['Time Occurred']%100)

"""
print data16.info()

data16['timeAndDate'] = ''
#print data16.head(10)
count = 0
for i,row in data16.iterrows():
	count += 1
	if count%10000 == 0:
		print count
	if math.isnan(row['Time Occurred']) is True:
		pass
		#print count
		#data16.drop(i)
	else:
		#print 'inside'
		temp = row['Date Occurred'] + datetime.timedelta(hours = math.floor(row['Time Occurred']/100), minutes = math.floor(row['Time Occurred'])%100)
		#print i,temp
		data16.set_value(i,'timeAndDate',temp)

data16 = data16[0:278176]
#print data16.head(20)

print data16.info()

data16.to_pickle('data16_1.pkl')
print('------------------------------------------------------------========================================================================================')


x = raw_input('wait')
"""

"""
data16 = pd.read_pickle('data16_1.pkl')
print data16.info()
data16.sort_values(by = 'timeAndDate',ascending = True, inplace = True)

print data16.head(20)
data16.to_pickle('data16_sortedDateTime.pkl')
"""
#x = raw_input()
rows = int(sys.argv[1])
cols = int(sys.argv[2])

#if os.path.isfile('matrix_' + str(rows) + 'x' + str(cols)+'.npy'):
#	sys.exit()
	#sys.exit('makeGrid.py not needed as grid is already saved')

data16 = pd.read_pickle('../crimePrediction/data16_sortedDateTime.pkl')
#data16.info()
#data16.head()
#numpy.set_printoptions(threshold='nan')

#plt.hist(data16['Crime Code Desc'])
#objects = data16['Crime Code Desc'].unique()
#x_pos = numpy.arrange(len(objects))

#pd.set_option('display.max_rows', len(data16['Crime Code Desc'].value_counts()))
#print data16['Crime Code Desc'].value_counts()
#pd.reset_option('display.max_rows')
#freq = [data16['Crime Code Desc'] for object in objects]


#plt.show()
#aw_input('')
#count = 0

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




matrix = [[[] for x in range(cols)] for y in range(rows)]
#matrix_violent = [[[] for x in range(cols)] for y in range(rows)]
#print matrix

#list = matrix[0][0]
#list.append(3)

height = math.fabs(ur_lat - ll_lat)/rows
width = math.fabs(ur_lon - ll_lon)/cols

#print height,width

count = 0
n = 0

for each in data16.iterrows():
	#t = each[1]['Date Occurred']
	#print each[1]['Date Occurred']
	#print 'time'
	#print each[1]['timeAndDate']
	n += 1 

	if n%10000 == 0:
		#print n
		pass

	locString = each[1]['Location ']

	#print locString[1:-1].split(',')
	arr = locString[1:-1].split(',')

	lon,lat = float(arr[1]),float(arr[0])
	#print lon,lat
	
	if isInsideBox(ll_lon,ll_lat,ur_lon,ur_lat,lon,lat) == True:
		count += 1
		row_index = int(math.floor((ur_lat - lat)/height))
		col_index = int(math.floor((lon - ll_lon)/width))
		#print row_index,col_index

		crimeCode = each[1]['Crime Code']

		if int(crimeCode) in lightRelatedCrimes :
			list = matrix[row_index][col_index]
		else:
			pass
			#list = matrix_violent[row_index][col_index]
		
		list.append(each[1]['timeAndDate'])

numpy.save('matrix_' + str(rows) + 'x' + str(cols) + '_LAPD_area_lightCrime',matrix)
#numpy.save('violent_matrix_' + str(rows) + 'x' + str(cols) + '_LAPD_area',matrix_violent)
#print count