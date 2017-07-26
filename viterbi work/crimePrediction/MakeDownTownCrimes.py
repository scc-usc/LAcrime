"""
Divide crime data into grid cells and save in matrix format
Author : Omkar Damle
Date : 5th June 2017
"""

import pandas as pd
import math
import privateInfo
from pyproj import Proj, transform
import datetime
import numpy
import sys
import os.path

def isInsideBox(ll_lon,ll_lat,ur_lon,ur_lat,lon,lat):
	isInside = True
	
	if lat > ur_lat or lat < ll_lat or lon > ur_lon or lon < ll_lon:
		isInside = False

	return isInside



LA_COORDINATES = (34.05, -118.24)
#crimedata = pd.read_csv('lapd_Crime_Data_From_2010_to_Present.csv')
#08/30/2010,2300
#crimedata['Date Occurred'] = pd.to_datetime(crimedata['Date Occurred'],format='%m/%d/%Y')
#crimedata.sort_values(by='Date Occurred',ascending = True, inplace = True)
# for speed purposes
MAX_RECORDS = 10


data16 = pd.read_pickle('data16_sortedDateTime.pkl')
#data16.info()
#data16.head()

count = 0

#Downtown LA
# Lower Left -> 34.038811, -118.273534
# Upper Right -> 34.053781, -118.237727

#horizontal length = 3.27km
#vertical length = 1.74km

ll_lat = 34.038811
ll_lon = -118.273534
ur_lat = 34.053781
ur_lon = -118.237727


count = 0
n = 0

dtCrime = []

for each in data16.iterrows():
	#t = each[1]['Date Occurred']
	#print each[1]['Date Occurred']
	#print 'time'
	#print each[1]['timeAndDate']
	n += 1 

	if n%10000 == 0:
		print n
	crimeTime = each[1]['timeAndDate']
	locString = each[1]['Location ']

	#print locString[1:-1].split(',')
	arr = locString[1:-1].split(',')

	lon,lat = float(arr[1]),float(arr[0])
	#print lon,lat
	
	if isInsideBox(ll_lon,ll_lat,ur_lon,ur_lat,lon,lat) == True:
		count += 1
		dtCrime.append((lon,lat,crimeTime))

numpy.save('downTownCrimesLocations',dtCrime)
