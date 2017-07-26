"""
LA atm locations
Author : Omkar Damle
Date : 22nd June 2017
"""
import pandas as pd
import math
from pyproj import Proj, transform
import numpy

def isInsideBox(ll_lon,ll_lat,ur_lon,ur_lat,lon,lat):
	isInside = True
	
	if lat > ur_lat or lat < ll_lat or lon > ur_lon or lon < ll_lon:
		isInside = False

	return isInside

streetData = pd.read_csv('LADOT_Meters_Citywide_Dec_2015.csv')

# for speed purposes
MAX_RECORDS = 10000
LA_COORDINATES = (34.05, -118.24)

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
downTownMeters = []

for each in streetData.iterrows():
	count += 1
	if count%10000 == 0:
		print count
	arr1 = each[1]['the_geom'][7:-1].split()
	lon1,lat1 = float(arr1[0]), float(arr1[1])		
		
	if isInsideBox(ll_lon,ll_lat,ur_lon,ur_lat,lon1,lat1) == True:

		if math.isnan(lat1) or math.isnan(lon1):
			pass
		else:
			downTownMeters.append((lon1,lat1))
			
numpy.save('downTownMeters',downTownMeters)
print(len(downTownMeters))
